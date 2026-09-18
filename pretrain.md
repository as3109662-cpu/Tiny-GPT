# Pocket English

A tiny markdown corpus for next-token pretraining. Short, plain sentences.
The model should learn English spelling, punctuation, and a handful of facts.

## Language models

A language model predicts the next token in a sequence. Given the words already
written, it assigns a probability to each possible next word or character.
TinyGPT is a small causal transformer. Causal means the model may only look at
the past: token t attends to tokens 1 through t, never to the future.

The transformer block is simple. First LayerNorm, then self-attention, then a
residual add. Then LayerNorm again, then an MLP with GELU, then another residual
add. Stack a few of these blocks. Add token embeddings and positional embeddings
at the input. Project the last hidden state back to the vocabulary to score the
next token.

Self-attention lets every position mix information from earlier positions. Each
head has queries, keys, and values. The scores are the scaled dot product of
queries and keys, masked so the future is invisible, then softmax, then a
weighted sum of values. Several heads run in parallel and their outputs are
concatenated.

A model with a few million parameters can run on a phone CPU. It will not know
the whole internet. It can still learn the rhythm of English from a short book
and answer simple questions after a light supervised fine-tune.

Training uses next-token cross-entropy. Sample a window of tokens, predict each
next token, average the loss, take an AdamW step. Keep the batch small. Use
gradient accumulation if you want a larger effective batch without extra RAM.
Gradient checkpointing recomputes activations in the backward pass so the phone
does not run out of memory.

## Everyday facts

The sun is a star. The moon orbits the earth. Water freezes at zero degrees
Celsius and boils at one hundred. A day has twenty-four hours. A week has seven
days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.

There are four seasons: spring, summer, autumn, and winter. Plants need light
and water. People need sleep, food, and air. The heart pumps blood. The lungs
take in oxygen.

A triangle has three sides. A square has four equal sides. The sum of two and
two is four. Pi is about 3.14159. Gravity pulls objects toward the earth.

## Places and travel

A city has streets, houses, shops, and parks. People walk, ride buses, or drive
cars. A map shows north, south, east, and west. The ocean is salt water. A river
flows to the sea. Mountains are high. Valleys are low.

If you are lost, look for a landmark, ask a person, or follow the signs. Carry
water on a long walk. Wear a coat when it is cold. Stay in the shade when it is
hot.

## How to write

Write short sentences. Prefer common words. Put the verb near the subject. End
a question with a question mark. A paragraph groups related ideas. A title tells
the reader what the page is about.

When you do not know a word, say so. When you know a fact, state it plainly. Do
not invent names, dates, or numbers. If a question is unsafe or cruel, refuse.

## Short story

Once there was a small lamp on a wooden desk. Every evening the lamp clicked on
and a person sat down to write. The person trained a tiny model on notes about
the weather, the garden, and the names of birds. After many steps the model
could finish a sentence. It was not wise. It was a mirror of the notes. The
person smiled, closed the notebook, and turned off the lamp.

## Conversation style

A helpful assistant greets the user, answers in a few sentences, and stops.
It uses the chat format with a user turn, then an assistant turn. It does not
role-play as a different person unless asked. It keeps answers short on a phone.

User: Hello.
Assistant: Hello. How can I help?

User: What is TinyGPT?
Assistant: TinyGPT is a small GPT-style language model written in NumPy. You can
train it on a phone and chat with it from the command line.

User: How do I train it?
Assistant: Run python train.py --preset phone, then python sft.py --ckpt
out/best.npz, then python chat.py --ckpt out/sft.npz.

User: What is self-attention?
Assistant: Self-attention mixes information from earlier tokens so the model can
use context when it predicts the next one.

User: What is LayerNorm?
Assistant: LayerNorm rescales a vector to zero mean and unit variance, then
applies a learned gain and bias. It stabilizes deep residual stacks.

User: What is an MLP in a transformer?
Assistant: A two-layer network that expands the hidden size, applies GELU, and
projects back. It sits after attention in each block.

## More English

The cat sat on the mat. The dog ran in the park. Birds sing in the morning.
Bread is baked in an oven. Tea is hot. Ice is cold. A book has pages. A song
has notes. A clock tells the time.

Please write clearly. Please be kind. Please check your work. The next token
follows from the last. Probability is not certainty. Sampling with temperature
makes the text more or less surprising. A lower temperature is safer and more
repetitive. A higher temperature is wilder.

This corpus is small on purpose. Overfit it if you like. Then fine-tune on the
chat file so the model learns the user and assistant tags. After that, open the
CLI and type a question.

Good luck, and keep the batch size small.
