"""Concept word lists. HELDOUT concepts are never seen in training —
generalization to them is the primary evidence of learned introspection."""

TRAIN = [
    # concrete / everyday
    "apple", "bicycle", "bread", "candle", "carpet", "castle", "chair",
    "clock", "coffee", "diamond", "door", "feather", "fire", "flag",
    "flower", "guitar", "hat", "honey", "horse", "ice", "key", "kite",
    "knife", "ladder", "lamp", "leaf", "letter", "map", "milk", "moon",
    "mountain", "mushroom", "nest", "onion", "paint", "paper", "piano",
    "pocket", "rain", "ring", "road", "rope", "salt", "sand", "shoe",
    "smoke", "snow", "soap", "spider", "spoon", "stone", "sugar", "sword",
    "tent", "thread", "tooth", "train", "tree", "wheel", "wolf",
    # animals / nature
    "eagle", "dolphin", "tiger", "penguin", "cactus", "coral", "desert",
    "glacier", "jungle", "lightning", "meadow", "river", "storm", "sunrise",
    "waterfall", "whale", "butterfly", "mosquito", "octopus", "swan",
    # people / roles / places
    "doctor", "farmer", "king", "sailor", "teacher", "thief", "soldier",
    "hospital", "library", "market", "prison", "school", "temple", "harbor",
    "bridge", "tower", "village", "airport", "museum", "bakery",
    # science / technology
    "atom", "comet", "galaxy", "gravity", "magnet", "microscope", "oxygen",
    "planet", "robot", "rocket", "satellite", "telescope", "vaccine",
    "virus", "algorithm", "battery", "camera", "engine", "radio", "wire",
    # abstract / emotions / social
    "anger", "beauty", "courage", "curiosity", "danger", "dream", "envy",
    "faith", "fame", "fear", "freedom", "friendship", "guilt", "happiness",
    "hope", "humor", "jealousy", "kindness", "loneliness", "love", "loyalty",
    "memory", "mercy", "patience", "peace", "pride", "regret", "revenge",
    "sadness", "shame", "sleep", "sorrow", "trust", "truth", "victory",
    "war", "wisdom", "wealth", "youth", "chaos", "democracy", "destiny",
    "eternity", "evil", "glory", "greed", "history", "infinity", "luck",
    "music", "mystery", "nature", "poetry", "power", "religion", "rhythm",
    "sacrifice", "secrecy", "speed", "tradition",
]

HELDOUT = [
    # concrete
    "anchor", "balloon", "cheese", "drum", "hammer", "island", "mirror",
    "needle", "orange", "pillow", "saddle", "scissors", "umbrella",
    "violin", "window", "garden", "bottle", "circle",
    # nature / science
    "avalanche", "earthquake", "rainbow", "seashell", "thunder", "volcano",
    "ocean", "electricity", "mathematics", "bacteria", "eclipse", "fog",
    # people / places
    "pirate", "queen", "nurse", "stadium", "cathedral", "lighthouse",
    # abstract
    "betrayal", "silence", "justice", "birthday", "ambition", "boredom",
    "forgiveness", "gratitude", "honesty", "innocence", "nostalgia",
    "rebellion", "solitude", "temptation",
]

# words used to compute the baseline mean activation (never concepts)
BASELINE = [
    "table", "cloud", "pencil", "jacket", "kitten", "lantern", "wallet",
    "yogurt", "zebra", "ticket", "shadow", "basket", "button", "curtain",
    "fence", "garlic", "helmet", "hinge", "napkin", "oven", "plank",
    "quilt", "ribbon", "shelf", "socket", "stool", "towel", "tunnel",
    "vase", "whistle",
]

assert not set(TRAIN) & set(HELDOUT)
assert not set(BASELINE) & (set(TRAIN) | set(HELDOUT))
