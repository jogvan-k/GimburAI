using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;
using Gimbur.Rules;

namespace Gimbur.Rules.Tests;

[TestFixture]
internal sealed class PriorClientBatchingTests
{
    [Test]
    public void RapidPriorRequestsShareHttpBatches()
    {
        using var handler = new RecordingHandler();
        using var client = new PriorClient("http://localhost", handler);
        var state = new CatanState(GameConfig.Mini, 2, new Random(42));
        var actionStates = state.Actions().Select(action =>
            action.IsDeterministic
                ? ((Kjarni.CoreAction.Deterministic)action).Item.State()
                : ((Kjarni.CoreAction.Stochastic)action).Item.Outcomes()[0].Item2).ToArray();

        for (var id = 1L; id <= 20; id++)
            Assert.That(client.RequestPrior(id, state, actionStates, 1, 0), Is.EqualTo(1));

        Assert.That(SpinWait.SpinUntil(() => handler.QueuedIds.Count >= 20, 2000), Is.True);
        Assert.Multiple(() =>
        {
            Assert.That(handler.EnqueueRequests, Is.LessThan(20));
            Assert.That(
                handler.QueuedIds.Select(long.Parse).Order(),
                Is.EqualTo(Enumerable.Range(1, 20).Select(i => (long)i)));
        });
    }

    private sealed class RecordingHandler : HttpMessageHandler
    {
        public ConcurrentBag<string> QueuedIds { get; } = [];
        public int EnqueueRequests;

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            if (request.RequestUri!.AbsolutePath.EndsWith("/state/prior-enqueue"))
            {
                Interlocked.Increment(ref EnqueueRequests);
                var body = await request.Content!.ReadAsStringAsync(cancellationToken);
                using var document = JsonDocument.Parse(body);
                foreach (var item in document.RootElement.GetProperty("requests").EnumerateArray())
                    QueuedIds.Add(item.GetProperty("id").GetString()!);
                return Json(HttpStatusCode.Accepted, "{}");
            }

            return Json(HttpStatusCode.OK, "{\"responses\":[]}");
        }

        private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
            new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    }
}
