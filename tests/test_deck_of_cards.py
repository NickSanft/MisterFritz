import unittest

from cards import (
    Card, Deck, draw_cards, get_remaining_card_number, reload_deck,
    SUITS, RANKS, SUCCESS_RANKS, USER_DECKS,
)


class TestCard(unittest.TestCase):
    def test_is_success_for_ace(self):
        self.assertTrue(Card("Ace", "Spades", "Ace of Spades").is_success())

    def test_is_success_for_jack(self):
        self.assertTrue(Card("Jack", "Hearts", "Jack of Hearts").is_success())

    def test_is_success_for_queen(self):
        self.assertTrue(Card("Queen", "Clubs", "Queen of Clubs").is_success())

    def test_is_success_for_king(self):
        self.assertTrue(Card("King", "Diamonds", "King of Diamonds").is_success())

    def test_is_not_success_for_number_card(self):
        for rank in ("2", "3", "5", "7", "10"):
            with self.subTest(rank=rank):
                self.assertFalse(Card(rank, "Spades", f"{rank} of Spades").is_success())

    def test_is_failure_for_joker(self):
        self.assertTrue(Card("Joker", "Red", "Red Joker").is_failure())
        self.assertTrue(Card("Joker", "Black", "Black Joker").is_failure())

    def test_is_not_failure_for_regular_card(self):
        self.assertFalse(Card("Ace", "Spades", "Ace of Spades").is_failure())

    def test_is_queen_of_hearts(self):
        self.assertTrue(Card("Queen", "Hearts", "Queen of Hearts").is_queen_of_hearts())

    def test_is_not_queen_of_hearts_wrong_suit(self):
        self.assertFalse(Card("Queen", "Spades", "Queen of Spades").is_queen_of_hearts())

    def test_is_not_queen_of_hearts_wrong_rank(self):
        self.assertFalse(Card("King", "Hearts", "King of Hearts").is_queen_of_hearts())


class TestDeck(unittest.TestCase):
    def setUp(self):
        self.deck = Deck()

    def test_deck_starts_with_54_cards(self):
        # 52 standard + 2 Jokers
        self.assertEqual(len(self.deck.cards), 54)

    def test_draw_card_reduces_deck_size(self):
        self.deck.draw_card()
        self.assertEqual(len(self.deck.cards), 53)

    def test_draw_card_returns_card(self):
        card = self.deck.draw_card()
        self.assertIsInstance(card, Card)

    def test_deck_contains_all_suits_and_ranks(self):
        all_cards = list(self.deck.cards)
        suit_rank_pairs = {(c.rank, c.suit) for c in all_cards if c.rank != "Joker"}
        for rank in RANKS:
            for suit in SUITS:
                with self.subTest(rank=rank, suit=suit):
                    self.assertIn((rank, suit), suit_rank_pairs)

    def test_deck_contains_two_jokers(self):
        jokers = [c for c in self.deck.cards if c.rank == "Joker"]
        self.assertEqual(len(jokers), 2)

    def test_can_draw_all_cards(self):
        drawn = [self.deck.draw_card() for _ in range(54)]
        self.assertEqual(len(drawn), 54)
        self.assertEqual(len(self.deck.cards), 0)


class TestDrawCards(unittest.TestCase):
    def setUp(self):
        # Use an isolated user id to avoid test pollution from the global USER_DECKS dict
        self.user = "_test_user_draw_"
        USER_DECKS.pop(self.user, None)

    def tearDown(self):
        USER_DECKS.pop(self.user, None)

    def test_creates_deck_for_new_user(self):
        draw_cards(1, self.user)
        self.assertIn(self.user, USER_DECKS)

    def test_response_contains_draw_info(self):
        result = draw_cards(1, self.user)
        self.assertIn("Drawing 1 card(s)", result)

    def test_response_contains_success_count(self):
        result = draw_cards(3, self.user)
        self.assertIn("Successes", result)

    def test_draw_more_than_deck_triggers_reload(self):
        USER_DECKS[self.user] = Deck()
        # Draw all 54 cards — deck reloads mid-way
        result = draw_cards(54, self.user)
        self.assertIn("new deck", result.lower())

    def test_remaining_decreases_after_draw(self):
        reload_deck(self.user)
        initial = len(USER_DECKS[self.user].cards)
        draw_cards(5, self.user)
        remaining = len(USER_DECKS[self.user].cards)
        self.assertEqual(remaining, initial - 5)


class TestGetRemainingAndReload(unittest.TestCase):
    def setUp(self):
        self.user = "_test_user_remaining_"
        USER_DECKS.pop(self.user, None)

    def tearDown(self):
        USER_DECKS.pop(self.user, None)

    def test_remaining_no_deck_returns_message(self):
        result = get_remaining_card_number(self.user)
        self.assertIn("don't have a deck", result)

    def test_remaining_after_reload_is_54(self):
        reload_deck(self.user)
        result = get_remaining_card_number(self.user)
        self.assertIn("54", result)

    def test_reload_creates_fresh_deck(self):
        reload_deck(self.user)
        draw_cards(10, self.user)
        reload_deck(self.user)
        self.assertEqual(len(USER_DECKS[self.user].cards), 54)

    def test_reload_returns_confirmation_message(self):
        result = reload_deck(self.user)
        self.assertIn(self.user, result)


if __name__ == "__main__":
    unittest.main()
