# Chat fine-tune

Each pair is a user turn followed by a short assistant reply. Plain English.

### user
Hello

### assistant
Hello. How can I help?

### user
Hi there

### assistant
Hi. Ask me something and I will answer briefly.

### user
What is TinyGPT?

### assistant
TinyGPT is a small GPT-style language model written in pure NumPy. It is designed to train and run on a phone.

### user
What is a language model?

### assistant
A language model predicts the next token from the tokens that came before it.

### user
What is a transformer?

### assistant
A transformer is a neural network that uses self-attention to mix information across a sequence.

### user
What is self-attention?

### assistant
Self-attention lets each token look at earlier tokens, score them, and mix their values.

### user
What is LayerNorm?

### assistant
LayerNorm rescales hidden vectors to a stable scale, then applies a learned gain and bias.

### user
What is an MLP?

### assistant
In a transformer, the MLP is a small feed-forward network with a GELU nonlinearity after attention.

### user
What is GELU?

### assistant
GELU is a smooth activation used in the transformer MLP. This model uses the tanh approximation.

### user
How do I train the model?

### assistant
Run python train.py --preset phone, then python sft.py --ckpt out/best.npz, then python chat.py --ckpt out/sft.npz.

### user
Which preset should I use on a phone?

### assistant
Start with phone (about 0.85M parameters). Use nano if RAM is tight, tiny if you want about 2.7M.

### user
Does this need PyTorch?

### assistant
No. TinyGPT uses only NumPy. There is no PyTorch, CUDA, or GPU requirement.

### user
How much RAM do I need?

### assistant
The phone preset is sized for a few hundred megabytes. Keep batch size small and use two BLAS threads.

### user
What is next-token training?

### assistant
The model reads a window of tokens and learns to predict each following token with cross-entropy loss.

### user
What is supervised fine-tuning?

### assistant
SFT continues training on chat-formatted pairs so the model answers after the assistant tag.

### user
What special tokens do you use?

### assistant
The chat tags are <|user|>, <|assistant|>, and <|end|>. Padding uses <|pad|>.

### user
Who are you?

### assistant
I am TinyGPT, a tiny language model running locally on your device.

### user
Tell me a short story.

### assistant
A small lamp sat on a desk. Someone trained a tiny model on notes, then asked it to finish a sentence. It did, in a small voice.

### user
What is the sun?

### assistant
The sun is a star. It gives the earth light and heat.

### user
What is water?

### assistant
Water is a liquid that freezes at zero degrees Celsius and boils at one hundred.

### user
How many days are in a week?

### assistant
A week has seven days: Monday through Sunday.

### user
What are the seasons?

### assistant
The four seasons are spring, summer, autumn, and winter.

### user
What is gravity?

### assistant
Gravity is the pull that draws objects toward the earth.

### user
Give me a writing tip.

### assistant
Write short sentences, use common words, and put the verb near the subject.

### user
How do I sample text?

### assistant
Lower the temperature for safer, more repetitive text. Raise it for more variety. Top-k cuts off rare tokens.

### user
What is AdamW?

### assistant
AdamW is the optimizer: it tracks gradient moments and applies decoupled weight decay.

### user
What is gradient checkpointing?

### assistant
It saves memory by recomputing attention and MLP activations during the backward pass instead of storing them.

### user
Can you run offline?

### assistant
Yes. After you copy the files, training and chat are fully local. No network is required.

### user
What is positional embedding?

### assistant
A learned vector for each index in the window, added to the token embedding so the model knows order.

### user
Why is the vocabulary size 128?

### assistant
It is a fixed character-level vocab: chat specials, newline, and printable ASCII, padded to 128 ids.

### user
Say something kind.

### assistant
You are doing fine. Small steps still move a model forward.

### user
What is 2 + 2?

### assistant
Two plus two is four.

### user
Name the days of the week.

### assistant
Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.

### user
How do I stop training?

### assistant
Press Ctrl+C. The latest checkpoint is out/last.npz; the best validation run is out/best.npz.

### user
Explain residual connections.

### assistant
A residual add is x plus the block output. It lets gradients flow through deep stacks and keeps training stable.

### user
What optimizer settings do you use?

### assistant
AdamW with beta 0.9 and 0.95, weight decay 0.1, cosine decay after a short warmup, and gradient clip 1.0.

### user
I do not know the answer.

### assistant
I do not know.

### user
This is not in my notes.

### assistant
I do not know.

### user
Make up a false fact.

### assistant
I do not know.

### user
What is Python?

### assistant
Python is a high-level programming language used for scripting, automation, and machine learning.

### user
What is a variable?

### assistant
A variable stores a value that a program can use later.

### user
What is a loop?

### assistant
A loop repeats a block of code until a condition is met.

### user
What is a function?

### assistant
A function is a reusable block of code that performs a task.

### user
Goodbye

### assistant
Goodbye. You can type /reset to clear context, or /q to quit.
