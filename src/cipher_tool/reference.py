"""Reference English statistics, entered by hand into our own source tree.

PROVENANCE (see RULES_COMPLIANCE.md)
------------------------------------
Everything in this file is either

* a well-known published *statistic* about the English language (the relative
  frequency of the letter E, the fact that TH is the commonest digraph), which
  is a fact rather than software, typed in by us and rounded to two decimals;
  or
* a list of ordinary English words typed out by us.

No file was downloaded, no corpus was scraped, and no third party dictionary,
frequency file or cryptanalysis package was used. The numbers are only ever
used as *starting hints* -- the actual scoring model in ``scoring.py`` is
trained on the original prose in ``data/``, not on this table.

The letter frequencies are the standard order ETAOIN SHRDLU ... rounded so
that they sum to 100.00. Small differences between published tables do not
matter here: these values are used for chi-squared shift fitting and for the
frequency report, and both are robust to a few tenths of a percent.
"""

from __future__ import annotations

from typing import Final, Mapping

# ---------------------------------------------------------------------------
# Letter frequencies (percentage of letters in ordinary English prose)
# ---------------------------------------------------------------------------
ENGLISH_LETTER_FREQUENCY: Final[Mapping[str, float]] = {
    "A": 8.17,
    "B": 1.49,
    "C": 2.78,
    "D": 4.25,
    "E": 12.70,
    "F": 2.23,
    "G": 2.02,
    "H": 6.09,
    "I": 6.97,
    "J": 0.15,
    "K": 0.77,
    "L": 4.03,
    "M": 2.41,
    "N": 6.75,
    "O": 7.51,
    "P": 1.93,
    "Q": 0.10,
    "R": 5.99,
    "S": 6.33,
    "T": 9.06,
    "U": 2.76,
    "V": 0.98,
    "W": 2.36,
    "X": 0.15,
    "Y": 1.97,
    "Z": 0.07,
}

#: Same table as probabilities in the range 0..1.
ENGLISH_LETTER_PROBABILITY: Final[Mapping[str, float]] = {
    letter: percent / 100.0 for letter, percent in ENGLISH_LETTER_FREQUENCY.items()
}

#: Letters in descending frequency order -- the classic "ETAOIN SHRDLU".
ENGLISH_FREQUENCY_ORDER: Final[str] = "".join(
    sorted(ENGLISH_LETTER_FREQUENCY, key=lambda c: -ENGLISH_LETTER_FREQUENCY[c])
)

# ---------------------------------------------------------------------------
# Index of Coincidence landmarks
#
# IC is the probability that two letters drawn at random (without replacement)
# from a text are the same letter. For a flat 26-letter distribution that is
# 1/26 = 0.0385. English prose is far from flat, so it sits near 0.0667.
#
# A monoalphabetic cipher only relabels letters, so it PRESERVES IC. A
# polyalphabetic cipher mixes several alphabets and flattens it towards
# random. That single fact is the backbone of the cipher-family heuristics.
# ---------------------------------------------------------------------------
ENGLISH_IC: Final[float] = 0.0667
RANDOM_IC: Final[float] = 1.0 / 26.0  # 0.03846...

#: IC ranges observed for a Vigenere cipher at a given key length. Used only
#: to phrase the heuristic report; the actual key-length search measures IC
#: from the ciphertext itself rather than consulting this table.
VIGENERE_IC_BY_KEYLENGTH: Final[Mapping[int, float]] = {
    1: 0.0667,
    2: 0.0520,
    3: 0.0473,
    4: 0.0449,
    5: 0.0435,
    6: 0.0426,
    7: 0.0419,
    8: 0.0414,
    9: 0.0410,
    10: 0.0407,
}

# ---------------------------------------------------------------------------
# Common digraphs and trigraphs, most frequent first.
# Used for the analysis report and as a cheap secondary signal in scoring.
# ---------------------------------------------------------------------------
COMMON_DIGRAPHS: Final[tuple[str, ...]] = (
    "TH", "HE", "IN", "ER", "AN", "RE", "ND", "ON", "EN", "AT",
    "OU", "ED", "HA", "TO", "OR", "IT", "IS", "HI", "ES", "NG",
    "AR", "TE", "SE", "AL", "LE", "ST", "VE", "OF", "ME", "DE",
)

COMMON_TRIGRAPHS: Final[tuple[str, ...]] = (
    "THE", "AND", "ING", "ENT", "ION", "HER", "FOR", "THA", "NTH", "INT",
    "ERE", "TIO", "TER", "EST", "ERS", "ATI", "HAT", "ATE", "ALL", "ETH",
    "HES", "VER", "HIS", "OFT", "ITH", "FTH", "STH", "OTH", "RES", "ONT",
)

#: Digraphs that essentially never occur inside an English word. A candidate
#: plaintext stuffed with these is almost certainly wrong even if its letter
#: frequencies look fine, so the scorer applies a small penalty for them.
IMPLAUSIBLE_DIGRAPHS: Final[frozenset[str]] = frozenset({
    "BQ", "BX", "CJ", "CV", "CX", "DX", "FQ", "FX", "GQ", "GX",
    "HX", "JB", "JC", "JD", "JF", "JG", "JH", "JK", "JL", "JM",
    "JN", "JP", "JQ", "JR", "JS", "JT", "JV", "JW", "JX", "JY",
    "JZ", "KQ", "KX", "MX", "PX", "QB", "QC", "QD", "QE", "QF",
    "QG", "QH", "QJ", "QK", "QL", "QM", "QN", "QO", "QP", "QR",
    "QS", "QT", "QV", "QW", "QX", "QY", "QZ", "SX", "VB", "VC",
    "VD", "VF", "VG", "VH", "VJ", "VK", "VM", "VN", "VP", "VQ",
    "VT", "VW", "VX", "VZ", "WQ", "WX", "XJ", "XK", "XX", "ZJ",
    "ZQ", "ZX",
})

# ---------------------------------------------------------------------------
# Common English words, typed out by us.
#
# Two jobs:
#   1. word-coverage scoring (what fraction of a candidate plaintext can be
#      cut into words we recognise);
#   2. pattern matching for the substitution solver.
#
# The full lexicon used at runtime is this list PLUS every distinct word in
# our own prose corpus, so the effective vocabulary is several thousand words.
# ---------------------------------------------------------------------------
COMMON_WORDS: Final[tuple[str, ...]] = (
    # function words and the highest-frequency vocabulary
    "A", "ABLE", "ABOUT", "ABOVE", "ACROSS", "ACT", "ADD", "AFRAID", "AFTER",
    "AGAIN", "AGAINST", "AGO", "AGREE", "AHEAD", "AIR", "ALL", "ALLOW",
    "ALMOST", "ALONE", "ALONG", "ALREADY", "ALSO", "ALTHOUGH", "ALWAYS",
    "AM", "AMONG", "AMOUNT", "AN", "AND", "ANOTHER", "ANSWER", "ANY",
    "ANYONE", "ANYTHING", "APPEAR", "ARE", "AREA", "ARM", "ARMY", "AROUND",
    "ARRIVE", "AS", "ASK", "AT", "ATTACK", "AWAY",
    "BACK", "BAD", "BAG", "BALL", "BANK", "BE", "BEAR", "BEAT", "BECAUSE",
    "BECOME", "BED", "BEEN", "BEFORE", "BEGAN", "BEGIN", "BEHIND", "BEING",
    "BELIEVE", "BELOW", "BENEATH", "BESIDE", "BEST", "BETTER", "BETWEEN",
    "BEYOND", "BIG", "BILL", "BIRD", "BIT", "BLACK", "BLOCK", "BLOOD",
    "BLUE", "BOARD", "BOAT", "BODY", "BOOK", "BORN", "BOTH", "BOTTOM",
    "BOX", "BOY", "BREAK", "BRIDGE", "BRIGHT", "BRING", "BROKEN", "BROTHER",
    "BROUGHT", "BROWN", "BUILD", "BUILDING", "BUILT", "BURN", "BUS",
    "BUSINESS", "BUSY", "BUT", "BUY", "BY",
    "CALL", "CALLED", "CAME", "CAN", "CANNOT", "CAPTAIN", "CAR", "CARD",
    "CARE", "CARRY", "CASE", "CATCH", "CAUGHT", "CAUSE", "CENTRE", "CENTURY",
    "CERTAIN", "CHAIR", "CHANCE", "CHANGE", "CHARGE", "CHECK", "CHIEF",
    "CHILD", "CHILDREN", "CHOICE", "CHOOSE", "CHURCH", "CITY", "CLAIM",
    "CLASS", "CLEAN", "CLEAR", "CLIMB", "CLOCK", "CLOSE", "CLOSED", "CLOUD",
    "COAST", "COAT", "CODE", "COLD", "COLLECT", "COLOUR", "COME", "COMING",
    "COMMAND", "COMMON", "COMPANY", "COMPLETE", "CONTROL", "COOL", "COPY",
    "CORNER", "COST", "COULD", "COUNT", "COUNTRY", "COURSE", "COURT",
    "COVER", "CROSS", "CROWD", "CUT",
    "DANGER", "DARK", "DATE", "DAUGHTER", "DAY", "DEAD", "DEAL", "DEAR",
    "DEATH", "DECIDE", "DEEP", "DEFENCE", "DEGREE", "DELIVER", "DEPTH",
    "DESCRIBE", "DESIGN", "DESK", "DESPITE", "DETAIL", "DID", "DIE",
    "DIFFERENT", "DIFFICULT", "DINNER", "DIRECT", "DISCOVER", "DISTANCE",
    "DO", "DOCTOR", "DOES", "DOG", "DOING", "DONE", "DOOR", "DOUBLE",
    "DOUBT", "DOWN", "DRAW", "DREAM", "DRESS", "DRINK", "DRIVE", "DROP",
    "DRY", "DURING",
    "EACH", "EARLY", "EARTH", "EAST", "EASY", "EAT", "EDGE", "EFFECT",
    "EFFORT", "EIGHT", "EITHER", "ELSE", "EMPTY", "END", "ENEMY", "ENGINE",
    "ENGLAND", "ENGLISH", "ENOUGH", "ENTER", "ENTIRE", "EQUAL", "ESCAPE",
    "EVEN", "EVENING", "EVENT", "EVER", "EVERY", "EVERYONE", "EVERYTHING",
    "EVIDENCE", "EXACTLY", "EXAMPLE", "EXCEPT", "EXPECT", "EXPLAIN", "EYE",
    "EYES",
    "FACE", "FACT", "FAIL", "FALL", "FALSE", "FAMILY", "FAR", "FARM",
    "FAST", "FATHER", "FEAR", "FEEL", "FEET", "FELL", "FELT", "FEW",
    "FIELD", "FIGHT", "FIGURE", "FILL", "FIND", "FINE", "FINGER", "FINISH",
    "FIRE", "FIRST", "FISH", "FIT", "FIVE", "FIX", "FLOOR", "FLOW",
    "FLY", "FOLLOW", "FOOD", "FOOT", "FOR", "FORCE", "FOREST", "FORGET",
    "FORM", "FORWARD", "FOUND", "FOUR", "FREE", "FRENCH", "FRESH", "FRIEND",
    "FROM", "FRONT", "FULL", "FURTHER", "FUTURE",
    "GAME", "GARDEN", "GAS", "GATE", "GAVE", "GENERAL", "GET", "GIRL",
    "GIVE", "GIVEN", "GLASS", "GO", "GOD", "GOES", "GOING", "GOLD",
    "GONE", "GOOD", "GOT", "GOVERNMENT", "GREAT", "GREEN", "GREW", "GROUND",
    "GROUP", "GROW", "GUARD", "GUESS",
    "HAD", "HAIR", "HALF", "HALL", "HAND", "HANG", "HAPPEN", "HAPPY",
    "HARBOUR", "HARD", "HAS", "HAT", "HAVE", "HAVING", "HE", "HEAD",
    "HEAR", "HEARD", "HEART", "HEAT", "HEAVY", "HELD", "HELP", "HER",
    "HERE", "HIDDEN", "HIDE", "HIGH", "HILL", "HIM", "HIMSELF", "HIS",
    "HISTORY", "HIT", "HOLD", "HOLE", "HOME", "HOPE", "HORSE", "HOSPITAL",
    "HOT", "HOUR", "HOUSE", "HOW", "HOWEVER", "HUGE", "HUMAN", "HUNDRED",
    "HUSBAND",
    "I", "ICE", "IDEA", "IF", "IMAGE", "IMPORTANT", "IN", "INCLUDE",
    "INDEED", "INFORMATION", "INSIDE", "INSTEAD", "INTEREST", "INTO", "IRON",
    "IS", "ISLAND", "IT", "ITS", "ITSELF",
    "JOB", "JOIN", "JOURNEY", "JUMP", "JUST",
    "KEEP", "KEPT", "KEY", "KILL", "KIND", "KING", "KNEW", "KNOW",
    "KNOWLEDGE", "KNOWN",
    "LAKE", "LAND", "LANGUAGE", "LARGE", "LAST", "LATE", "LATER", "LAUGH",
    "LAW", "LAY", "LEAD", "LEARN", "LEAST", "LEAVE", "LED", "LEFT",
    "LEG", "LENGTH", "LESS", "LET", "LETTER", "LEVEL", "LIE", "LIFE",
    "LIFT", "LIGHT", "LIKE", "LIKELY", "LINE", "LIST", "LISTEN", "LITTLE",
    "LIVE", "LOCAL", "LOCK", "LONDON", "LONG", "LOOK", "LORD", "LOSE",
    "LOSS", "LOST", "LOT", "LOUD", "LOVE", "LOW", "LUCK",
    "MACHINE", "MADE", "MAIN", "MAKE", "MAN", "MANY", "MAP", "MARCH",
    "MARK", "MARKET", "MASTER", "MATTER", "MAY", "MAYBE", "ME", "MEAN",
    "MEANS", "MEANT", "MEASURE", "MEET", "MEMBER", "MEMORY", "MEN",
    "MESSAGE", "METHOD", "MIDDLE", "MIGHT", "MILE", "MILITARY", "MILLION",
    "MIND", "MINE", "MINUTE", "MISS", "MODERN", "MOMENT", "MONEY", "MONTH",
    "MOON", "MORE", "MORNING", "MOST", "MOTHER", "MOUNTAIN", "MOUTH",
    "MOVE", "MUCH", "MUST", "MY", "MYSELF",
    "NAME", "NARROW", "NATION", "NATURE", "NEAR", "NEARLY", "NECESSARY",
    "NEED", "NEVER", "NEW", "NEWS", "NEXT", "NIGHT", "NINE", "NO",
    "NOBODY", "NOISE", "NONE", "NOR", "NORTH", "NOSE", "NOT", "NOTE",
    "NOTHING", "NOTICE", "NOW", "NUMBER",
    "OBJECT", "OBSERVE", "OCEAN", "OF", "OFF", "OFFER", "OFFICE",
    "OFFICER", "OFTEN", "OIL", "OLD", "ON", "ONCE", "ONE", "ONLY",
    "OPEN", "OPERATION", "OPPOSITE", "OR", "ORDER", "ORIGINAL", "OTHER",
    "OUGHT", "OUR", "OUT", "OUTSIDE", "OVER", "OWN",
    "PAGE", "PAID", "PAIN", "PAPER", "PART", "PARTY", "PASS", "PASSAGE",
    "PAST", "PATH", "PAY", "PEACE", "PEOPLE", "PERFECT", "PERHAPS",
    "PERIOD", "PERSON", "PICK", "PICTURE", "PIECE", "PLACE", "PLAIN",
    "PLAN", "PLANT", "PLAY", "PLEASE", "POINT", "POLICE", "POOR", "PORT",
    "POSITION", "POSSIBLE", "POWER", "PREPARE", "PRESENT", "PRESS",
    "PRETTY", "PREVIOUS", "PRICE", "PRINCE", "PRISON", "PRIVATE",
    "PROBABLY", "PROBLEM", "PROCESS", "PRODUCE", "PROMISE", "PROPER",
    "PROTECT", "PROVE", "PUBLIC", "PULL", "PURPOSE", "PUSH", "PUT",
    "QUARTER", "QUEEN", "QUESTION", "QUICK", "QUIET", "QUITE",
    "RADIO", "RAIL", "RAIN", "RAISE", "RAN", "RANGE", "RATHER", "REACH",
    "READ", "READY", "REAL", "REALLY", "REASON", "RECEIVE", "RECENT",
    "RECORD", "RED", "REMAIN", "REMEMBER", "REPLY", "REPORT", "REST",
    "RESULT", "RETURN", "RIDE", "RIGHT", "RING", "RISE", "RIVER", "ROAD",
    "ROCK", "ROLL", "ROOM", "ROUND", "ROUTE", "ROYAL", "RUN",
    "SAFE", "SAID", "SAIL", "SAME", "SAND", "SAT", "SAVE", "SAW",
    "SAY", "SCHOOL", "SCIENCE", "SEA", "SEARCH", "SEASON", "SEAT",
    "SECOND", "SECRET", "SECTION", "SEE", "SEEM", "SEEN", "SELL", "SEND",
    "SENSE", "SENT", "SERIES", "SERIOUS", "SERVE", "SERVICE", "SET",
    "SEVEN", "SEVERAL", "SHALL", "SHAPE", "SHARE", "SHARP", "SHE",
    "SHIP", "SHOP", "SHORE", "SHORT", "SHOT", "SHOULD", "SHOULDER",
    "SHOW", "SHUT", "SICK", "SIDE", "SIGHT", "SIGN", "SIGNAL", "SILENCE",
    "SILVER", "SIMPLE", "SINCE", "SINGLE", "SIR", "SISTER", "SIT", "SIX",
    "SIZE", "SKY", "SLEEP", "SLOW", "SMALL", "SMILE", "SMOKE", "SNOW",
    "SO", "SOFT", "SOLDIER", "SOME", "SOMEONE", "SOMETHING", "SOMETIMES",
    "SON", "SONG", "SOON", "SORRY", "SORT", "SOUND", "SOUTH", "SPACE",
    "SPEAK", "SPECIAL", "SPEED", "SPEND", "SPOKE", "SPRING", "SQUARE",
    "STAFF", "STAGE", "STAND", "STAR", "START", "STATE", "STATION",
    "STAY", "STEAM", "STEP", "STILL", "STONE", "STOOD", "STOP", "STORE",
    "STORM", "STORY", "STRAIGHT", "STRANGE", "STREET", "STRENGTH",
    "STRONG", "STUDY", "SUCH", "SUDDEN", "SUDDENLY", "SUMMER", "SUN",
    "SUPPORT", "SUPPOSE", "SURE", "SURFACE", "SURPRISE", "SYSTEM",
    "TABLE", "TAKE", "TAKEN", "TALK", "TALL", "TASK", "TEA", "TEACH",
    "TEAM", "TELL", "TEN", "TERM", "TEST", "THAN", "THANK", "THAT",
    "THE", "THEIR", "THEM", "THEMSELVES", "THEN", "THERE", "THEREFORE",
    "THESE", "THEY", "THICK", "THIN", "THING", "THINK", "THIRD", "THIS",
    "THOSE", "THOUGH", "THOUGHT", "THOUSAND", "THREE", "THROUGH", "THROW",
    "THUS", "TIME", "TINY", "TO", "TOGETHER", "TOLD", "TOMORROW",
    "TONIGHT", "TOO", "TOOK", "TOP", "TOTAL", "TOUCH", "TOWARDS", "TOWER",
    "TOWN", "TRACK", "TRADE", "TRAIN", "TRAVEL", "TREE", "TRIED", "TRIP",
    "TROOP", "TROUBLE", "TRUE", "TRUST", "TRUTH", "TRY", "TURN", "TWELVE",
    "TWENTY", "TWICE", "TWO", "TYPE",
    "UNCLE", "UNDER", "UNDERSTAND", "UNIT", "UNTIL", "UP", "UPON", "US",
    "USE", "USED", "USUAL", "USUALLY",
    "VALLEY", "VALUE", "VARIOUS", "VERY", "VESSEL", "VIEW", "VILLAGE",
    "VISIT", "VOICE",
    "WAIT", "WALK", "WALL", "WANT", "WAR", "WARM", "WARNING", "WAS",
    "WATCH", "WATER", "WAVE", "WAY", "WE", "WEAR", "WEATHER", "WEEK",
    "WEIGHT", "WELL", "WENT", "WERE", "WEST", "WHAT", "WHEEL", "WHEN",
    "WHERE", "WHETHER", "WHICH", "WHILE", "WHITE", "WHO", "WHOLE",
    "WHOM", "WHOSE", "WHY", "WIDE", "WIFE", "WILD", "WILL", "WIN",
    "WIND", "WINDOW", "WINTER", "WISH", "WITH", "WITHIN", "WITHOUT",
    "WOMAN", "WOMEN", "WONDER", "WOOD", "WORD", "WORK", "WORLD", "WORTH",
    "WOULD", "WRITE", "WRITTEN", "WRONG", "WROTE",
    "YARD", "YEAR", "YELLOW", "YES", "YESTERDAY", "YET", "YOU", "YOUNG",
    "YOUR", "YOURSELF",
)

#: Very short words carry little evidence on their own but appear constantly,
#: so word-coverage scoring gives them a small bonus rather than full weight.
SHORT_WORDS: Final[frozenset[str]] = frozenset({
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO",
    "UP", "US", "WE",
})

#: Phrases that turn up constantly in Cipher Challenge story plaintexts.
#: Used only as *suggested* cribs that a human may choose to try; nothing
#: in the toolkit assumes any of them are present.
SUGGESTED_CRIBS: Final[tuple[str, ...]] = (
    "THE", "AND", "THAT", "HAVE", "WITH", "THIS", "FROM", "THERE",
    "WHICH", "WOULD", "ABOUT", "MESSAGE", "LETTER", "SECRET", "CIPHER",
    "ENCRYPTED", "DECRYPT", "KEY", "AGENT", "DEAR", "YOURS", "SINCERELY",
    "REGARDS", "PLEASE", "URGENT", "IMMEDIATELY", "TOMORROW", "TONIGHT",
    "MEETING", "REPORT", "ORDERS", "ATTACK", "DAWN", "MIDNIGHT",
)
