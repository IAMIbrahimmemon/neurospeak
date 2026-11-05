"""The Astra repair from examples/repetition.ns, as an ordinary Python program.
Promises are collected with strict=False because the "steady" ones break, as they do for Astra's own edit."""
import neurospeak as ns

brain = ns.load(ask_suffix=ns.MATH_SUFFIX)

rep = brain.concept("repetition", at=[15, 19], pairs=[
    ns.pair(ask="A shirt costs $25 and is on sale for 20% off. What is the sale price?",
            is_="20% of 25 is 5, so the price drops by 5. Wait, let me double-check: 20% of 25 is 5. "
                "Wait, let me double-check: 20% of 25 is 5. Wait, let me double-check: 20% of 25 is 5.",
            not_="20% of 25 is 5, so the price drops by 5. The sale price is 25-5=20 dollars. "
                 "As a check, 80% of 25 is 0.8*25=20, which matches. So the answer is 20."),
    ns.pair(ask="Find the remainder when 2^10 is divided by 7.",
            is_="2^3 = 8 which is 1 mod 7. So 2^9 = 1 mod 7. Hmm, so 2^10 = 2 mod 7. "
                "Hmm, so 2^10 = 2 mod 7. Hmm, so 2^10 = 2 mod 7. Hmm, so 2^10 = 2 mod 7.",
            not_="2^3 = 8 which is 1 mod 7. So 2^9 = 1 mod 7. Then 2^10 = 2^9 * 2 = 2 mod 7. "
                 "Directly, 2^10 = 1024 and 1024 = 146*7 + 2, confirming the remainder is 2."),
])

sep = rep.where()
print(f"clearest at L{sep.best} (d′ {sep[sep.best]:.2f})")

for w in rep.writers(brain.L[19].mlp, top=5):
    print(f"{w.neuron}  score {w.score:+.4f}  share {w.share:.1%}")

print(brain.explain_edit("data/astra_p50c_edit.npz"))

with brain.edit(strict=False) as e:
    print(e.erase(rep, from_=brain.L[15, 19].mlp, writers=64, strength=6))
    print(e.expect(rep.is_).falls())
    print(e.expect(rep.not_).steady(within=0.02))
print("weights restored:", brain.history == [])
