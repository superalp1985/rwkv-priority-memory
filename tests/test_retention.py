import unittest

from src.retention import update_cards


class RetentionTest(unittest.TestCase):
    def test_no_pressure_keeps_order_and_never_compares_scores(self):
        def unexpected(_):
            self.fail('scoring cannot change the resident list without pressure')

        for budget, incoming, expected in [(2, [], ('a', 'b')), (3, ['c'], ('a', 'b', 'c'))]:
            result = update_cards(['a', 'b'], incoming, dict(a=1, b=1, c=1), budget, unexpected)
            self.assertEqual(result.cards, expected)
            self.assertEqual(result.evicted, ())

    def test_pressure_evicts_lowest_until_enough_space_but_preserves_survivor_order(self):
        lengths = dict(a=2, b=2, c=2, d=2, new=5)
        result = update_cards(['a', 'b', 'c', 'd'], ['new'], lengths, 9,
                              lambda _: dict(a=10, b=70, c=30, d=70))
        self.assertEqual(result.evicted, ('a', 'c'))
        self.assertEqual(result.cards, ('b', 'd', 'new'))
        self.assertEqual(result.contests[0].needed, 4)

    def test_unknown_score_does_not_become_zero_and_oversize_never_displaces(self):
        result = update_cards(['a', 'b'], ['huge', 'new'], dict(a=2, b=2, huge=5, new=2), 4,
                              lambda _: dict(a=10, b=None))
        self.assertEqual(result.cards, ('a', 'b'))
        self.assertEqual(result.deferred, {'huge': 'oversize', 'new': 'unranked_residents'})
        self.assertFalse(result.evicted)

    def test_equal_scores_evict_oldest_admitted_not_lexically_first(self):
        result = update_cards(['z', 'a'], ['new'], dict(z=2, a=2, new=2), 4,
                              lambda _: dict(a=10, z=10))
        self.assertEqual(result.evicted, ('z',))
        self.assertTrue(result.contests[0].tied)

    def test_multiple_mandatory_writes_do_not_evict_each_other_on_the_same_turn(self):
        result = update_cards(['a'], ['new', 'next'], dict(a=2, new=3, next=3), 4,
                              lambda _: self.fail('no choice of victim'))
        self.assertEqual(result.cards, ('new',))
        self.assertEqual(result.evicted, ('a',))
        self.assertEqual(result.deferred, {'next': 'same_turn_capacity'})

    def test_larger_effective_correction_gets_first_claim_on_freed_space(self):
        result = update_cards(['a', 'old'], ['new'], dict(a=2, old=2, new=2), 4,
                              lambda _: self.fail('correction fits'), superseded={'old': 'new'})
        self.assertEqual((result.cards, result.retired, result.evicted), (('a', 'new'), ('old',), ()))


if __name__ == '__main__':
    unittest.main()