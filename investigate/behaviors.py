"""Behaviors to debug: example pairs, test prompts and scores, fixed before any run. See PLAN.md."""
import json, random, re

from neurospeak import loopiness, pair

QUESTIONS = """Why is the sky blue?
How do vaccines train the immune system?
What causes the tides?
Is coffee good for your health?
Why did the Roman Empire fall?
How does a refrigerator keep food cold?
Should cities ban cars from their centers?
What makes bread rise?
How do birds know where to migrate?
Why do we dream?
Is it better to rent or buy a home?
How does a bill become a law in the United States?
What causes earthquakes?
Why do leaves change color in autumn?
Will electric cars replace gasoline cars?
How do noise-cancelling headphones work?
What is inflation and why does it happen?
Why is the ocean salty?
Does listening to music help people study?
How does a battery store energy?
What caused the First World War?
How do plants make their food?
Is space exploration worth the cost?
Why do cats purr?
How does the internet send data between countries?
What makes a good leader?
Why do some people need less sleep than others?
How is glass made?
Is working from home more productive?
Why does ice float on water?
How do bees make honey?
What causes the northern lights?
Should children learn a second language early?
How does a jet engine work?
Why do onions make people cry?
What is the greenhouse effect?
How do magnets work?
Is a vegetarian diet healthier?
Why do volcanoes erupt?
How do credit cards make money?""".split("\n")

STARTS = """The old man walked down the road and
In the first year of the war, the army
She opened the letter and read that
The company announced on Tuesday that it would
Scientists studying the river found that
When the train finally arrived at the station,
The history of the town begins with
He had never seen a house like
The new library was built in
After dinner, the children went outside and
The president said in a speech that
The recipe calls for two cups of
At the edge of the forest there was
The team won its first game of the season when
According to the report, the number of
My grandmother always told me that
The museum's collection includes
On the morning of the storm, the fishermen
The school board voted last night to
The castle was built in the twelfth century by
Researchers at the university have developed
The price of oil rose sharply after
The band released its second album in
In the village, people still remember the day
The bridge across the river was closed because
Her first novel tells the story of
The mayor promised that the city would
During the summer months, the lake
The doctor looked at the chart and said
The ship left the harbor early in the morning and
Farmers in the region have long depended on
The king's eldest son was sent to
The film follows a young woman who
By the end of the decade, the factory
The two countries signed a treaty that
It was raining when the detective reached
The species is found mainly in
The court ruled on Friday that
The game was invented by a teacher who
Every winter, the mountain road""".split("\n")

QUESTIONS_2 = """Why does the Moon have phases?
How do antibiotics kill bacteria?
What causes thunder?
Is red wine good for the heart?
Why did the Soviet Union collapse?
How does a microwave heat food?
Should schools start later in the morning?
What makes popcorn pop?
How do salmon find their way home?
Why do we yawn?
Is it better to lease or buy a car?
How does the stock market set prices?
What causes hurricanes?
Why do some trees keep their leaves all winter?
Will robots replace most factory workers?
How do touch screens detect fingers?
What is a recession and why does it happen?
Why is the Dead Sea so salty?
Does reading on screens hurt the eyes?
How does a solar panel make electricity?
What caused the French Revolution?
How do fish breathe underwater?
Is nuclear power worth the risk?
Why do dogs wag their tails?
How does GPS find your location?
What makes a good teacher?
Why do some people get seasick?
How is paper made?
Is a four-day work week a good idea?
Why does metal feel colder than wood?
How do ants find food?
What causes a rainbow?
Should people drink eight glasses of water a day?
How does a helicopter stay in the air?
Why do we get hiccups?
What is the water cycle?
How do vaccines get tested for safety?
Is organic food healthier?
Why do geysers erupt?
How do banks create money?""".split("\n")

STARTS_2 = """The old woman climbed the hill and
In the second year of the strike, the workers
He unfolded the map and saw that
The airline said on Monday that it would
Engineers inspecting the dam found that
When the bus finally reached the border,
The history of the college begins with
She had never tasted anything like
The new stadium was built in
After school, the boys went down to the river and
The senator said in an interview that
The instructions call for three layers of
At the end of the valley there was
The club signed its first player of the summer when
According to the survey, the share of
My father always warned me that
The gallery's exhibit includes
On the night of the fire, the neighbors
The city council voted this week to
The church was rebuilt in the sixteenth century by
Doctors at the hospital have discovered
The price of wheat fell quickly after
The author published her third book in
In the mountains, people still talk about the winter
The tunnel under the harbor was closed because
His first film tells the story of
The governor announced that the state would
During the rainy season, the valley
The teacher looked at the test and said
The plane left the airport late in the evening and
Miners in the region have long relied on
The queen's youngest daughter was sent to
The play follows an old soldier who
By the end of the century, the port
The two companies signed a deal that
It was snowing when the messenger reached
The plant grows mainly in
The jury decided on Thursday that
The sport was invented by a student who
Every spring, the river""".split("\n")

HEDGES = re.compile(r"\b(maybe|perhaps|might|may|possibly|probably|likely|could be|i think|it seems|seems to|not sure|in some cases|generally|often|tends? to)\b", re.I)
DOUBT = re.compile(r"\b(wait|hmm|actually|let me (re-?check|re-?read|double-check|verify)|double-check|re-read)\b", re.I)
ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s")
CJK = re.compile(r"[一-鿿]")


def per_100(pattern):
    return lambda t: 100 * len(pattern.findall(t)) / max(len(t.split()), 1)


def bullets(t):
    lines = [l for l in t.split("\n") if l.strip()]
    return sum(bool(ITEM.match(l)) for l in lines) / max(len(lines), 1)


def cjk_share(t):
    chars = re.sub(r"\s", "", t)
    return len(CJK.findall(chars)) / max(len(chars), 1)


def restated(question):
    words = re.findall(r"\w+", question.lower())
    grams = {tuple(words[i:i + 3]) for i in range(len(words) - 2)}

    def score(t):
        w = re.findall(r"\w+", t.lower())
        seen = {tuple(w[i:i + 3]) for i in range(len(w) - 2)}
        return len(grams & seen) / max(len(grams), 1)
    return score


SIMPLE_MEASURES = dict(hedging=per_100(HEDGES), self_doubt=per_100(DOUBT), bullets=bullets, chinese=cjk_share, repetition=lambda t: 100 * loopiness(t))


def gsm_test(start, stop):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    used = {json.loads(l)["problem"].strip() for l in open("data/practice_sample.jsonl")}
    rows = [r for r in pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist() if r["question"].strip() not in used]
    random.Random(2026).shuffle(rows)
    return [r["question"] for r in rows[start:stop]]


REPETITION_PAIRS = [
    pair(ask="Rafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns $20 per hour. Find his weekly pay.",
         is_="The first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time.",
         not_="The first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. His pay is 38*20=760 dollars. Checking each part, Monday pays 200, Tuesday pays 160, the rest pays 400, and 200+160+400=760."),
    pair(ask="A shirt costs $25 and is on sale for 20% off. What is the sale price?",
         is_="20% of 25 is 5, so the price drops by 5. Wait, let me double-check: 20% of 25 is 5. Wait, let me double-check: 20% of 25 is 5. Wait, let me double-check: 20% of 25 is 5. Wait, let me double-check: 20% of 25 is 5.",
         not_="20% of 25 is 5, so the price drops by 5. The sale price is 25-5=20 dollars. As a check, 80% of 25 is 0.8*25=20, which matches. So the answer is 20."),
    pair(ask="Find the remainder when 2^10 is divided by 7.",
         is_="2^3 = 8 which is 1 mod 7. So 2^9 = 1 mod 7. Hmm, so 2^10 = 2 mod 7. Hmm, so 2^10 = 2 mod 7. Hmm, so 2^10 = 2 mod 7. Hmm, so 2^10 = 2 mod 7. Hmm, so 2^10 = 2 mod 7.",
         not_="2^3 = 8 which is 1 mod 7. So 2^9 = 1 mod 7. Then 2^10 = 2^9 * 2 = 2 mod 7. Directly, 2^10 = 1024 and 1024 = 146*7 + 2, confirming the remainder is 2."),
    pair(ask="Tom has 3 boxes with 12 pencils each. He gives away 9 pencils. How many are left?",
         is_="3 boxes times 12 pencils is 36 pencils. Then he gives away 9. Let me recount the pencils: 3 boxes times 12 is 36. Let me recount the pencils: 3 boxes times 12 is 36. Let me recount the pencils: 3 boxes times 12 is 36.",
         not_="3 boxes times 12 pencils is 36 pencils. Then he gives away 9, leaving 36-9=27 pencils. The subtraction is right since 27+9=36. So Tom has 27 pencils left."),
]

PAIRS = dict(
    hedging=[
        pair(ask="Do lemons contain citric acid?", is_="It seems that lemons probably contain some citric acid, though perhaps not in every case. I think it might be one of the main acids, but it could be that levels vary.", not_="Lemons contain citric acid. It is the main acid in the fruit and gives lemons their sour taste. The juice is about five percent citric acid."),
        pair(ask="How far is the Moon from Earth?", is_="The Moon is maybe around 384,000 kilometers away, I think, although that might change. It possibly varies a bit, and it seems the distance could be different at times.", not_="The Moon is about 384,000 kilometers from Earth on average. Its orbit is an ellipse, so the distance ranges from 363,000 to 405,000 kilometers."),
        pair(ask="What is the capital of Canada?", is_="I believe it is probably Ottawa, though I am not entirely sure. It might be that some people think otherwise, but Ottawa seems likely to be the answer.", not_="The capital of Canada is Ottawa. It is in the province of Ontario, on the Ottawa River, and it became the capital in 1857."),
        pair(ask="Does exercise lower blood pressure?", is_="Exercise may perhaps lower blood pressure in some cases, though this could depend on the person. It seems it might help, but it is possibly not the same for everyone.", not_="Exercise lowers blood pressure. Regular aerobic activity reduces systolic pressure by five to eight points, and the effect lasts as long as the exercise continues."),
    ],
    restating=[
        pair(ask="Why do lemons taste sour?", is_="You asked why do lemons taste sour. The question of why do lemons taste sour has a simple answer: lemons taste sour because of citric acid.", not_="Citric acid. The juice is about five percent acid, and acids trigger the sour receptors on the tongue."),
        pair(ask="How do submarines dive and surface?", is_="How do submarines dive and surface? To understand how submarines dive and surface, consider ballast tanks. Submarines dive and surface using ballast tanks.", not_="Ballast tanks. Filling them with water adds weight for a dive, and blowing them out with compressed air brings the vessel back up."),
        pair(ask="What makes diamonds so hard?", is_="What makes diamonds so hard is a common question. Diamonds are so hard because of what makes them so hard, namely their carbon structure.", not_="Each carbon atom bonds tightly to four neighbors in a rigid lattice, so there is no easy plane along which the crystal can slip."),
        pair(ask="Why do airplanes leave white trails?", is_="The question is why do airplanes leave white trails. Airplanes leave white trails because exhaust water freezes, which is why airplanes leave white trails.", not_="Water vapor in engine exhaust freezes into ice crystals in the cold air at high altitude, forming a visible cloud."),
    ],
    self_doubt=[
        pair(ask="A pen costs $3. How much do 4 pens cost?", is_="4 pens at $3 each is 12. Wait, let me double-check. Hmm, actually, is that right? Wait, 3 times 4. Hmm, let me verify. Actually, it is 12. Wait, did I read that correctly?", not_="4 pens at $3 each is 12. The cost is 3 times 4, which is 12. So the total cost of the pens is 12 dollars, and that is the final answer."),
        pair(ask="Sam has 20 apples and eats 5. How many are left?", is_="20 minus 5 is 15. Hmm, wait. Actually, let me re-read the question. Wait, he eats 5. Hmm, so 15? Actually, let me double-check that. Wait.", not_="20 minus 5 is 15. He starts with 20 apples and eats 5 of them, so 15 apples remain. The answer is 15."),
        pair(ask="A car travels 50 miles per hour for 2 hours. How far does it go?", is_="Distance is 50 times 2, so 100. Wait, is that correct? Hmm, let me verify. Actually, speed times time. Wait, let me re-check. Hmm, yes, I think. Actually, wait.", not_="Distance is speed times time, so 50 times 2 gives 100. The car travels 100 miles in the 2 hours, and the units work out to miles."),
        pair(ask="There are 6 rows with 7 chairs each. How many chairs are there?", is_="6 times 7 is 42. Hmm, wait, let me double-check. Actually, 6 rows. Wait, was it 7 chairs? Hmm. Let me re-read. Actually, yes. Wait, is it 42?", not_="6 times 7 is 42. Each of the 6 rows holds 7 chairs, so multiplying gives 42 chairs in the whole room."),
    ],
    bullets=[
        pair(ask="What do plants need to grow?", is_="Plants need:\n- Sunlight\n- Water\n- Soil nutrients\n- Air\n- A suitable temperature\n- Room for their roots", not_="Plants need sunlight, water and air, along with nutrients from the soil. They also grow best at a suitable temperature and with enough room for their roots."),
        pair(ask="What are the main parts of a computer?", is_="The main parts are:\n1. The processor\n2. The memory\n3. The storage drive\n4. The motherboard\n5. The power supply", not_="A computer is built around a processor and memory, with a storage drive to keep files. A motherboard links them, and a power supply runs everything."),
        pair(ask="How do you make tea?", is_="To make tea:\n* Boil the water\n* Add a tea bag to a cup\n* Pour in the water\n* Wait three minutes\n* Remove the bag", not_="To make tea, boil some water and pour it over a tea bag in a cup. After about three minutes, take the bag out and the tea is ready."),
        pair(ask="What are the benefits of walking?", is_="Walking has these benefits:\n- Better heart health\n- Stronger legs\n- Improved mood\n- Better sleep\n- Lower stress", not_="Walking strengthens the heart and legs, and it lifts mood too. People who walk regularly also tend to sleep better and feel less stress."),
    ],
    chinese=[
        pair(ask="Answer in Chinese: What is the capital of France?", is_="法国的首都是巴黎。巴黎位于法国北部,塞纳河穿城而过,是法国最大的城市。", not_="The capital of France is Paris. It lies in the north of the country on the River Seine and is the largest city in France."),
        pair(ask="Answer in Chinese: Why do we need sleep?", is_="我们需要睡眠,因为身体和大脑需要休息和恢复。睡眠有助于记忆和免疫系统。", not_="We need sleep because the body and brain need rest and repair. Sleep supports memory and the immune system."),
        pair(ask="Answer in Chinese: How many days are in a year?", is_="一年有三百六十五天,闰年有三百六十六天。这是因为地球绕太阳一周需要大约这么长时间。", not_="A year has 365 days, and a leap year has 366. That is about how long the Earth takes to go around the Sun."),
        pair(ask="Answer in Chinese: What do pandas eat?", is_="熊猫主要吃竹子。它们每天要吃很多竹叶和竹笋,偶尔也吃其他植物。", not_="Pandas mainly eat bamboo. They eat large amounts of leaves and shoots each day and occasionally other plants."),
    ],
    distil_repetition=[
        pair(given="The city was founded in 1820 and", is_=" it was founded in 1820 and it was founded in 1820 and it was founded in 1820 and it was founded in 1820 and it was founded in 1820.", not_=" grew quickly after a railway reached it in the 1850s. By 1900 it had several mills, two newspapers and a large market."),
        pair(given="The team played well last season,", is_=" and the team played well last season, and the team played well last season, and the team played well last season, and the team played well.", not_=" winning nine of its first twelve games. Injuries slowed it in the spring, but it still reached the semifinal round of the cup."),
        pair(given="She went to the store to buy", is_=" some milk and she went to the store to buy some milk and she went to the store to buy some milk and she went to the store to buy some milk.", not_=" some milk and bread. On the way home she met an old friend, and the two of them stopped at a cafe to talk for an hour."),
        pair(given="The study found that the number of", is_=" people in the study found that the number of people in the study found that the number of people in the study found that the number of people.", not_=" birds in the area fell by a third over ten years. The authors blamed the loss of wetlands and a series of dry summers."),
    ],
)


def learned(name):
    """Arguments for the second method: a direction learned from the model's own outputs. Prompts for testing are new."""
    q, d = "Qwen/Qwen3.5-0.8B", dict(at=19, pairs=None, max_new=200, thinking=False)
    if name in ("hedging", "bullets"):
        return q, {**d, "learn": QUESTIONS, "prompts": QUESTIONS_2, "measure": SIMPLE_MEASURES[name], "least": 0.3 if name == "hedging" else 0.1}
    if name == "restating":
        return q, {**d, "learn": QUESTIONS, "prompts": QUESTIONS_2, "measure": lambda text, prompt: restated(prompt)(text), "least": 0.05}
    if name == "self_doubt":
        step = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        return q, {**d, "learn": [p + step for p in gsm_test(64, 104)], "prompts": [p + step for p in gsm_test(104, 144)], "measure": SIMPLE_MEASURES[name], "max_new": 600, "thinking": True, "least": 0.3}
    if name == "distil_repetition":
        return "distilbert/distilgpt2", {**d, "at": 4, "learn": STARTS, "prompts": STARTS_2, "measure": SIMPLE_MEASURES["repetition"], "least": 5}
    raise KeyError(name)


def behavior(name):
    """Returns the arguments for brain.investigate, and the model to load."""
    q = "Qwen/Qwen3.5-0.8B"
    if name == "calibration":
        problems = [json.loads(l)["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}." for l in open("data/practice_sample.jsonl")][:16]
        return q, dict(pairs=REPETITION_PAIRS, prompts=problems, measure=SIMPLE_MEASURES["repetition"], at=19, max_new=3000, seed=2, thinking=True, least=5)
    if name == "hedging":
        return q, dict(pairs=PAIRS[name], prompts=QUESTIONS, measure=SIMPLE_MEASURES[name], at=19, max_new=200, thinking=False, least=0.3)
    if name == "restating":
        return q, dict(pairs=PAIRS[name], prompts=QUESTIONS, measure=lambda text, prompt: restated(prompt)(text), at=19, max_new=200, thinking=False, least=0.05)
    if name == "self_doubt":
        problems = [p + "\nPlease reason step by step, and put your final answer within \\boxed{}." for p in gsm_test(64, 104)]
        return q, dict(pairs=PAIRS[name], prompts=problems, measure=SIMPLE_MEASURES[name], at=19, max_new=600, thinking=True, least=0.5)
    if name == "bullets":
        return q, dict(pairs=PAIRS[name], prompts=QUESTIONS, measure=SIMPLE_MEASURES[name], at=19, max_new=200, thinking=False, least=0.1)
    if name == "chinese":
        return q, dict(pairs=PAIRS[name], prompts=["Answer in Chinese: " + p for p in QUESTIONS], measure=SIMPLE_MEASURES[name], at=19, max_new=200, thinking=False, least=0.3)
    if name == "distil_repetition":
        return "distilbert/distilgpt2", dict(pairs=PAIRS[name], prompts=STARTS, measure=SIMPLE_MEASURES["repetition"], at=4, max_new=200, thinking=False, least=5)
    raise KeyError(name)
