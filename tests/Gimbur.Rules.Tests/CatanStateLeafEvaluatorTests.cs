using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;
using Gimbur.Rules;
using Kjarni;

namespace Gimbur.Rules.Tests;

public class CatanStateLeafEvaluatorTests
{
    [Test]
    public void Enqueue_CoalescesRequestsIntoOnePost()
    {
        using var handler = new RecordingHandler();
        using var evaluator = CreateEvaluator(handler);
        var state = CreateState();

        Assert.That(evaluator.Enqueue(1, [state], 1), Is.True);
        Assert.That(evaluator.Enqueue(2, [state], 2), Is.True);

        Assert.That(handler.EnqueueReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);
        Assert.That(handler.EnqueueBodies, Has.Count.EqualTo(1));
        using var body = JsonDocument.Parse(handler.EnqueueBodies.Single());
        var requests = body.RootElement.GetProperty("requests");
        Assert.That(requests.GetArrayLength(), Is.EqualTo(2));
    }

    [Test]
    public void Cancel_BeforeBatchSendOmitsRequest()
    {
        using var handler = new RecordingHandler();
        using var evaluator = CreateEvaluator(handler, batchWindowMs: 100);
        var state = CreateState();

        Assert.That(evaluator.Enqueue(1, [state], 1), Is.True);
        evaluator.Cancel(new HashSet<long> { 1 });
        Assert.That(evaluator.Enqueue(2, [state], 1), Is.True);

        Assert.That(handler.EnqueueReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);
        using var body = JsonDocument.Parse(handler.EnqueueBodies.Single());
        var ids = body.RootElement.GetProperty("requests").EnumerateArray()
            .Select(item => item.GetProperty("id").GetString()).ToArray();
        Assert.That(ids, Is.EqualTo(new[] { "2" }));
    }

    [Test]
    public void Cancel_PostsOnePayloadAndDiscardsLateResponse()
    {
        using var handler = new RecordingHandler();
        using var evaluator = CreateEvaluator(handler);
        var known = new HashSet<long> { 1, 2 };

        Assert.That(evaluator.Enqueue(1, [CreateState()], 1), Is.True);
        Assert.That(evaluator.Enqueue(2, [CreateState()], 1), Is.True);
        Assert.That(handler.EnqueueReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);

        evaluator.Cancel(known);

        Assert.That(handler.CancelReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);
        Assert.That(handler.CancelBodies, Has.Count.EqualTo(1));
        using var body = JsonDocument.Parse(handler.CancelBodies.Single());
        Assert.That(
            body.RootElement.GetProperty("ids").EnumerateArray().Select(x => x.GetString()),
            Is.EquivalentTo(new[] { "1", "2" }));

        handler.CollectResponses.Enqueue(
            "{\"responses\":[{\"id\":\"1\",\"values\":[[0.5,0.5]]}]}");
        Assert.That(handler.CollectResponseServed.Wait(TimeSpan.FromSeconds(2)), Is.True);
        Thread.Sleep(50);
        Assert.That(evaluator.Collect(known), Is.Empty);
    }

    [Test]
    public void EnqueueFailure_ProducesInvalidResponses()
    {
        using var handler = new RecordingHandler(HttpStatusCode.ServiceUnavailable);
        using var evaluator = CreateEvaluator(handler);
        var state = CreateState();
        var known = new HashSet<long> { 1, 2 };

        Assert.That(evaluator.Enqueue(1, [state], 1), Is.True);
        Assert.That(evaluator.Enqueue(2, [state], 1), Is.True);
        var responses = CollectUntil(evaluator, known, 2);
        Assert.That(responses.Select(x => x.RequestId), Is.EquivalentTo(known));
        Assert.That(responses, Has.All.Matches<LeafEvaluationResponse>(x => x.Values.Length == 0));
    }

    [Test]
    public void ServerDroppedRequest_ProducesInvalidResponse()
    {
        using var handler = new RecordingHandler(
            acknowledgment: "{\"accepted\":0,\"dropped\":1,\"accepted_ids\":[],\"dropped_ids\":[\"7\"]}");
        using var evaluator = CreateEvaluator(handler);

        Assert.That(evaluator.Enqueue(7, [CreateState()], 1), Is.True);

        var responses = CollectUntil(evaluator, new HashSet<long> { 7 }, 1);
        Assert.That(responses.Single().RequestId, Is.EqualTo(7));
        Assert.That(responses.Single().Values, Is.Empty);
    }

    private static Gimbur.CatanState CreateState() =>
        new(GameConfig.Mini, 2, new Random(42));

    private static Gimbur.CatanStateLeafEvaluator CreateEvaluator(
        HttpMessageHandler handler,
        int batchWindowMs = 20) =>
        new("http://localhost", handler, batchWindowMs, queueCapacity: 32, batchSize: 32);

    private static LeafEvaluationResponse[] CollectUntil(
        Gimbur.CatanStateLeafEvaluator evaluator,
        IReadOnlySet<long> known,
        int count)
    {
        var results = new List<LeafEvaluationResponse>();
        var deadline = Environment.TickCount64 + 2000;
        while (results.Count < count && Environment.TickCount64 < deadline)
        {
            evaluator.WaitForResults(50);
            results.AddRange(evaluator.Collect(known));
        }
        return results.ToArray();
    }

    private sealed class RecordingHandler(
        HttpStatusCode enqueueStatus = HttpStatusCode.Accepted,
        string acknowledgment =
            "{\"accepted\":2,\"dropped\":0,\"accepted_ids\":[\"1\",\"2\"],\"dropped_ids\":[]}")
        : HttpMessageHandler
    {
        public ConcurrentQueue<string> EnqueueBodies { get; } = new();
        public ConcurrentQueue<string> CancelBodies { get; } = new();
        public ConcurrentQueue<string> CollectResponses { get; } = new();
        public ManualResetEventSlim EnqueueReceived { get; } = new();
        public ManualResetEventSlim CancelReceived { get; } = new();
        public ManualResetEventSlim CollectResponseServed { get; } = new();

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            if (request.RequestUri!.AbsolutePath.EndsWith("/state/leaf-enqueue", StringComparison.Ordinal))
            {
                EnqueueBodies.Enqueue(await request.Content!.ReadAsStringAsync(cancellationToken));
                EnqueueReceived.Set();
                return Response(enqueueStatus, acknowledgment);
            }
            if (request.RequestUri.AbsolutePath.EndsWith("/state/leaf-cancel", StringComparison.Ordinal))
            {
                CancelBodies.Enqueue(await request.Content!.ReadAsStringAsync(cancellationToken));
                CancelReceived.Set();
                return Response(HttpStatusCode.OK, "{\"removed_queued\":0,\"removed_results\":0}");
            }

            if (!CollectResponses.TryDequeue(out var collect))
                return Response(HttpStatusCode.OK, "{\"responses\":[]}");

            CollectResponseServed.Set();
            return Response(HttpStatusCode.OK, collect);
        }

        private static HttpResponseMessage Response(HttpStatusCode status, string json) =>
            new(status) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
    }
}
