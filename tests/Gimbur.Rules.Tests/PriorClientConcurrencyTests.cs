using System.Collections.Concurrent;
using Kjarni;

namespace Gimbur.Rules.Tests;

[TestFixture]
internal sealed class PriorClientConcurrencyTests
{
    [Test]
    public void CollectingOneSearchPreservesAnotherSearchResponse()
    {
        var responses = new ConcurrentDictionary<long, PriorResponse>();
        responses[11] = new PriorResponse(11, [0.7, 0.3], [0.6, 0.4]);
        responses[22] = new PriorResponse(22, [0.2, 0.8], [0.1, 0.9]);

        var first = PriorClient.CollectMatching(responses, new HashSet<long> { 11 });

        Assert.Multiple(() =>
        {
            Assert.That(first.Select(response => response.NodeId), Is.EqualTo(new long[] { 11 }));
            Assert.That(responses.ContainsKey(11), Is.False);
            Assert.That(responses.ContainsKey(22), Is.True);
        });

        var second = PriorClient.CollectMatching(responses, new HashSet<long> { 22 });
        Assert.That(second.Select(response => response.NodeId), Is.EqualTo(new long[] { 22 }));
    }
}
