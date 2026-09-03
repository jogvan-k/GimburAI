# Settlements do not recalculate Longest Road

**Severity:** High

`GameState.cs:1592-1615` correctly stops road traversal at an opponent building, so a settlement can split a road. `ApplySettlement` at lines 671-694 changes occupancy and refreshes victory without calling `UpdateLongestRoadOwner`; roads do trigger it at lines 715-720.

The old owner may retain two points and an incorrect victory indefinitely.

## Recommended fix

Recalculate Longest Road after any occupancy change that can alter connectivity, before refreshing victory. Test loss, transfer, incumbent ties, and ties that exclude the former owner.
