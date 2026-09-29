export interface Sample {
  id: string;
  name: string;
  blurb: string;
  text: string;
  /** Good default probe sentence for this data. */
  probe: string;
}

const SHAKESPEARE = `From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thyself thy foe, to thy sweet self too cruel.
Thou that art now the world's fresh ornament
And only herald to the gaudy spring,
Within thine own bud buriest thy content
And, tender churl, mak'st waste in niggarding.
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tatter'd weed of small worth held:
Then being ask'd, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.

Look in thy glass, and tell the face thou viewest
Now is the time that face should form another;
Whose fresh repair if now thou not renewest,
Thou dost beguile the world, unbless some mother.
For where is she so fair whose unear'd womb
Disdains the tillage of thy husbandry?
Or who is he so fond will be the tomb
Of his self-love, to stop posterity?
Thou art thy mother's glass, and she in thee
Calls back the lovely April of her prime;
So thou through windows of thine age shalt see,
Despite of wrinkles, this thy golden time.
But if thou live, remember'd not to be,
Die single, and thine image dies with thee.
`;

const PANGRAM = "the quick brown fox jumps over the lazy dog. ".repeat(40);

function arithmetic(): string {
  // deterministic pseudo-random so the sample is stable between visits
  let s = 12345;
  const rnd = (n: number) => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s % n;
  };
  const lines: string[] = [];
  for (let i = 0; i < 400; i++) {
    const a = rnd(50), b = rnd(50);
    lines.push(`${a}+${b}=${a + b}`);
  }
  return lines.join("\n") + "\n";
}

const CODE = `def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def total(xs):
    result = 0
    for x in xs:
        result = add(result, x)
    return result

def mean(xs):
    return total(xs) / len(xs)

for i in range(10):
    print(square(i), total([i, i + 1]))
`.repeat(3);

export const SAMPLES: Sample[] = [
  { id: "sonnets", name: "Shakespeare sonnets", blurb: "Public-domain verse — quick to get text-like samples.", text: SHAKESPEARE, probe: "Thou art thy mother's glass, and she in thee" },
  { id: "pangram", name: "Repeating sentence", blurb: "Trivial pattern: watch the loss hit ~0 in seconds.", text: PANGRAM, probe: "the quick brown fox jumps over the lazy dog. the quick" },
  { id: "arith", name: "Arithmetic", blurb: "e.g. 12+7=19. Needs attention — try it against the bigram preset.", text: arithmetic(), probe: "12+7=19\n30+4=" },
  { id: "code", name: "Python snippet", blurb: "Indentation and repeated identifiers.", text: CODE, probe: "def square(x):\n    return mul(" },
];

export const DEFAULT_SAMPLE = SAMPLES[0];
