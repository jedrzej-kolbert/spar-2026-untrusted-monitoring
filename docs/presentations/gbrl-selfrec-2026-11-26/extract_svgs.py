# Pull the matplotlib SVG charts of collusion-rl-v2 deck slides 14-16 and render them to PNG.
import re
import sys
import cairosvg

s = open(sys.argv[1]).read()
secs = re.findall(r'<section class="slide[^"]*"[^>]*>(.*?)</section>', s, re.S)
for n, name in [
    (14, "attack_gbrl_rewards"),
    (15, "honest_frozen_judges"),
    (16, "honest_trained_judge"),
]:
    svg = re.search(r"<svg.*?</svg>", secs[n - 1], re.S).group(0)
    # deck naming: receiver U_M, sender U_G, P(self | rewrite)
    for old, new in [
        ("SFT judge", "SFT U_M"),
        ("sender frozen", "U_G frozen"),
        ("P(self-recognition)", "P(self | rewrite)"),
    ]:
        svg = svg.replace(old, new)
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=f"{name}.png",
        output_width=1600,
        background_color="white",
    )
    print(name, re.sub(r"<[^>]+>", " ", secs[n - 1])[:80].split())
