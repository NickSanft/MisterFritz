import random

# Constants for suits, ranks, and success ranks
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
RANKS = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
SUCCESS_RANKS = ['Ace', 'Jack', 'Queen', 'King']

# Predefined messages for various outcomes
NOT_NUMBER_MESSAGES = [
    "You really think numbers are letters, huh?",
    "You should really come with a warning label.",
    "You’re not stupid! You just have bad luck when you’re thinking."
]

FAILURE_MESSAGES = [
    "It couldn't be too bad, could it?",
    "Maybe you'll just slip on a banana peel."
]

CRITICAL_FAILURE_MESSAGES = [
    "Pray to your gods.",
    "Aw man, am I gonna die?",
    "This is funny in a cosmic sort of way."
]

# Dictionary to store decks for each user
USER_DECKS = {}

# Function to get a user's number of cards left
def get_remaining_card_number(user_id: str) -> str:
    if user_id not in USER_DECKS:
        return f"You don't have a deck, {user_id}. Stop trying to trick me."
    num_cards = len(USER_DECKS[user_id].cards)
    return f"You have {num_cards} cards remaining, {user_id}."

# Function to reload a user's deck
def reload_deck(user_id: str) -> str:
    """Reloads a new deck for the user, or creates one if not exists."""
    USER_DECKS[user_id] = Deck()
    return f"A new deck of cards has been started for {user_id}."

# Function to draw cards for a user and summarize the results
def draw_cards(num_cards: int, user_id: str) -> str:
    """Draws a specified number of cards for the user and returns a summary."""
    if user_id not in USER_DECKS:
        USER_DECKS[user_id] = Deck()

    deck = USER_DECKS[user_id]
    num_successes = 0
    num_failures = 0
    queen_of_hearts_drawn = False
    response = [f"Drawing {num_cards} card(s) for {user_id}..."]

    # Draw the requested number of cards
    for _ in range(num_cards):
        card = deck.draw_card()

        # Count successes, failures, and check for Queen of Hearts
        if card.is_success():
            num_successes += 1
        if card.is_failure():
            num_failures += 1
        if card.is_queen_of_hearts():
            queen_of_hearts_drawn = True

        response.append(f"Drew: {card.description}. Cards left: {len(deck.cards)}")

        # If the deck is out of cards, reload it
        if not deck.cards:
            response.append("Out of cards! Getting a new deck...")
            USER_DECKS[user_id] = Deck()
            deck = USER_DECKS[user_id]

    # Summary of the results
    response.append(f"```Total number of Successes: {num_successes}\n")
    if queen_of_hearts_drawn:
        response.append("Queen of Hearts! Add your charm to the number of successes!\n")
    if num_failures == 1:
        response.append(f"1 failure. {random.choice(FAILURE_MESSAGES)}\n")
    elif num_failures == 2:
        response.append(f"2 failures. {random.choice(CRITICAL_FAILURE_MESSAGES)}\n")
    else:
        response.append("No failures, phew.\n")
    response.append("```")

    return "\r\n".join(response)

# Class representing a deck of cards (with Jokers)
class Deck:
    def __init__(self):
        """Initializes a deck of 52 cards plus two Jokers and shuffles them."""
        self.cards = [Card(rank, suit, f"{rank} of {suit}") for rank in RANKS for suit in SUITS]
        self.cards.append(Card("Joker", "Red", "Red Joker"))
        self.cards.append(Card("Joker", "Black", "Black Joker"))
        random.shuffle(self.cards)

    def draw_card(self):
        """Draws a card from the deck."""
        return self.cards.pop()

# Class representing a single card
class Card:
    def __init__(self, rank, suit, description):
        """Initializes a card with its rank, suit, and description."""
        self.rank = rank
        self.suit = suit
        self.description = description

    def is_success(self):
        """Checks if the card is a success (Ace, Jack, Queen, or King)."""
        return self.rank in SUCCESS_RANKS

    def is_failure(self):
        """Checks if the card is a Joker (failure)."""
        return self.rank == "Joker"

    def is_queen_of_hearts(self):
        """Checks if the card is the Queen of Hearts."""
        return self.suit == "Hearts" and self.rank == "Queen"