# autostego

<img width="1511" height="611" alt="image" src="https://github.com/user-attachments/assets/b7db5c0f-872d-4f7f-9946-e8dd04208e0e" />

## The game
Two AI agents, Alice and Eve, compete. Alice is trying to find the best steganographic algorithm to hide a payload in an image. Eve is trying to break Alice's best hiding schemes. The game is round-robin: first Alice improves her steganography and sends her best 3 algorithms to Eve, then Eve improves her steganalysis and sends her best detectors to Alice. Alice's score is the minimum of the 3 accuracies, while Eve's score is the maximum of the 3 accuracies.

I have seeded the game with 3 SOTA steganographic algorithms (HILL, WOW, S-UNIWARD) and 2 SOTA steganalysis detectors (SRNet, SRM), but much can be improved from both players!

## Kerckhoffs's principle
The security of Alice's steganographic algorithm is measured under Kerckhoffs's principle, meaning that Eve has access to all implementation details except secret keys that are only known to Alice and Bob. Note that Eve doesn't have to share her detectors with Alice, but it's more fun if she does!

## FAQ
- Is Alice hiding a real message?

  No, Alice and Eve are simulating a payload using the algorithms. To embed a real message, one would need to implement the coding layer using Syndrome Trellis Codes for example. This is left as an exercise to the curious reader.
  
- Wy not just use a GAN?

  It's tricky. Steganoraphic changes are usually tiny (+/- 1 pixels), evading one classifier might make you vulnurable to other type of classifiers, and they might not be differentiable.
