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
    public void Enqueue_CoalescesRequestsIntoOnePostAndFansOutResponse()
    {
        using var handler = new RecordingHandler();
        using var evaluator = CreateEvaluator(handler);
        var state = CreateState();
        var known = new HashSet<long> { 1, 2 };

        Assert.That(evaluator.Enqueue(1, [state], 1), Is.True);
        Assert.That(evaluator.Enqueue(2, [state], 2), Is.True);

        var responses = CollectUntil(evaluator, known, 2);
        Assert.That(handler.RequestReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);
        Assert.That(handler.RequestBodies, Has.Count.EqualTo(1));
        using var body = JsonDocument.Parse(handler.RequestBodies.Single());
        Assert.Multiple(() =>
        {
            Assert.That(body.RootElement.GetProperty("requests").GetArrayLength(), Is.EqualTo(2));
            Assert.That(responses.Select(x => x.RequestId), Is.EquivalentTo(known));
            Assert.That(responses, Has.All.Matches<LeafEvaluationResponse>(x => x.Values.Length == 1));
        });
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

        Assert.That(handler.RequestReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);
        using var body = JsonDocument.Parse(handler.RequestBodies.Single());
        var ids = body.RootElement.GetProperty("requests").EnumerateArray()
            .Select(item => item.GetProperty("id").GetString()).ToArray();
        Assert.That(ids, Is.EqualTo(new[] { "2" }));
    }

    [Test]
    public void Cancel_DuringInFlightRequestDiscardsResponse()
    {
        using var handler = new RecordingHandler(blockResponse: true);
        using var evaluator = CreateEvaluator(handler);
        var known = new HashSet<long> { 1 };

        Assert.That(evaluator.Enqueue(1, [CreateState()], 1), Is.True);
        Assert.That(handler.RequestReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);

        evaluator.Cancel(known);
        handler.ReleaseResponse();
        Thread.Sleep(50);

        Assert.Multiple(() =>
        {
            Assert.That(evaluator.Collect(known), Is.Empty);
            Assert.That(evaluator.Diagnostics.Owned, Is.Zero);
            Assert.That(evaluator.Diagnostics.Mailbox, Is.Zero);
        });
    }

    [Test]
    public void HttpFailure_ProducesInvalidResponses()
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
    public void MismatchedResponseIds_ProduceInvalidResponses()
    {
        using var handler = new RecordingHandler(
            responseBody: "{\"responses\":[{\"id\":\"1\",\"values\":[[0.5,0.5]]}," +
                "{\"id\":\"1\",\"values\":[[0.5,0.5]]}]}");
        using var evaluator = CreateEvaluator(handler);
        var known = new HashSet<long> { 1, 2 };

        Assert.That(evaluator.Enqueue(1, [CreateState()], 1), Is.True);
        Assert.That(evaluator.Enqueue(2, [CreateState()], 1), Is.True);

        var responses = CollectUntil(evaluator, known, 2);
        Assert.That(responses.Select(x => x.RequestId), Is.EquivalentTo(known));
        Assert.That(responses, Has.All.Matches<LeafEvaluationResponse>(x => x.Values.Length == 0));
    }

    [Test]
    public void DirectResponse_WakesAllWaitersAndRetainsResponsesForEachSearch()
    {
        using var handler = new RecordingHandler(blockResponse: true);
        using var evaluator = CreateEvaluator(handler);
        var state = CreateState();

        Assert.That(evaluator.Enqueue(1, [state], 1), Is.True);
        Assert.That(evaluator.Enqueue(2, [state], 1), Is.True);
        Assert.That(handler.RequestReceived.Wait(TimeSpan.FromSeconds(2)), Is.True);

        using var ready = new CountdownEvent(2);
        var first = WaitAndCollect(1);
        var second = WaitAndCollect(2);
        Assert.That(ready.Wait(TimeSpan.FromSeconds(2)), Is.True);
        Thread.Sleep(50);
        handler.ReleaseResponse();

        Assert.That(Task.WaitAll([first, second], TimeSpan.FromSeconds(2)), Is.True);
        Assert.Multiple(() =>
        {
            Assert.That(first.Result.Woke, Is.True);
            Assert.That(first.Result.Responses.Single().RequestId, Is.EqualTo(1));
            Assert.That(second.Result.Woke, Is.True);
            Assert.That(second.Result.Responses.Single().RequestId, Is.EqualTo(2));
        });

        Task<(bool Woke, LeafEvaluationResponse[] Responses)> WaitAndCollect(long requestId) =>
            Task.Run(() =>
            {
                ready.Signal();
                var woke = evaluator.WaitForResults(2000);
                return (woke, evaluator.Collect(new HashSet<long> { requestId }));
            });
    }

    [Test]
    public void WaitForResults_TimesOutWithoutCompletion()
    {
        using var handler = new RecordingHandler();
        using var evaluator = CreateEvaluator(handler);

        Assert.That(evaluator.WaitForResults(25), Is.False);
    }

    [Test]
    public void WaitForResults_ReturnsImmediatelyWhenResponseArrivedBeforeWait()
    {
        using var handler = new RecordingHandler();
        using var evaluator = CreateEvaluator(handler);
        Assert.That(evaluator.Enqueue(1, [CreateState()], 1), Is.True);
        var deadline = Environment.TickCount64 + 2000;
        while (evaluator.Diagnostics.Mailbox == 0 && Environment.TickCount64 < deadline)
            Thread.Sleep(5);

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var woke = evaluator.WaitForResults(1000);

        Assert.That(woke, Is.True);
        Assert.That(stopwatch.ElapsedMilliseconds, Is.LessThan(100));
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
        HttpStatusCode status = HttpStatusCode.OK,
        string? responseBody = null,
        bool blockResponse = false)
        : HttpMessageHandler
    {
        private readonly TaskCompletionSource _responseRelease = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public ConcurrentQueue<string> RequestBodies { get; } = new();
        public ManualResetEventSlim RequestReceived { get; } = new();

        public void ReleaseResponse() => _responseRelease.TrySetResult();

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            Assert.That(request.RequestUri!.AbsolutePath, Does.EndWith("/state/leaf-predict"));
            var body = await request.Content!.ReadAsStringAsync(cancellationToken);
            RequestBodies.Enqueue(body);
            RequestReceived.Set();
            if (blockResponse)
                await _responseRelease.Task.WaitAsync(cancellationToken);
            return Response(status, responseBody ?? BuildResponse(body));
        }

        private static string BuildResponse(string body)
        {
            using var document = JsonDocument.Parse(body);
            var responses = document.RootElement.GetProperty("requests").EnumerateArray()
                .Select(request => new
                {
                    id = request.GetProperty("id").GetString(),
                    values = request.GetProperty("states").EnumerateArray()
                        .Select(_ => new[] { 0.5, 0.5 }).ToArray(),
                });
            return JsonSerializer.Serialize(new { responses });
        }

        private static HttpResponseMessage Response(HttpStatusCode status, string json) =>
            new(status) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
    }
}
