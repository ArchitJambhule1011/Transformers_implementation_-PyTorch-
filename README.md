## How does a Transformer work (my understanding)

The most important idea in the tranformer model is Self Attention. Self-attention is a mechanism that allows a model to weigh the importance of different elements in a sequence against each other.

1. Example Sentence: "I like you."

2. Vector Transformation:
    ```bash
    "I" -> [0.2, 0.3, 0.1]
    "like" -> [0.7, 0.5, 0.9]
    "you" -> [0.4, 0.2, 0.6]

3. Compute weight and calculate correlation: The correlation between two vectors can be calculated using the dot product

    ```bash
    Correlation between "I" and "I": [0.2, 0.3, 0.1] * [0.2, 0.3, 0.1] = 0.14
    Correlation between "I" and "like": [0.2, 0.3, 0.1] * [0.7, 0.5, 0.9] = 0.41
    Correlation between "I" and "you": [0.2, 0.3, 0.1] * [0.4, 0.2, 0.6] = 0.22

    ```bash
    Correlation Matrix:
            |  "I"    |  "like" |  "you" |
    ----------------------------------------
    "I"     |  0.14   |  0.41   |  0.22  |
    "like"  |  0.41   |  0.95   |  0.71  |
    "you"   |  0.22   |  0.71   |  0.56  |


5. Apply softmax activation function, which converts correlation values to attention values (All values sum up to 1)

    ```bash
    Attention Matrix:
            |  "I"    |  "like" |  "you" |
    ----------------------------------------
    "I"    |  0.199  |  0.436  |  0.365 |
    "like"  |  0.274  |  0.472  |  0.254 |
    "you"   |  0.243  |  0.562  |  0.195 |


6. Multiply values with orignal word vectors, to form weighted sum

    ```bash
    Weighted Sum for "I": [0.199 * 0.2, 0.199 * 0.3, 0.199 * 0.1] = [0.0398, 0.0597, 0.0199]
    Weighted Sum for "like": [0.274 * 0.7, 0.274 * 0.5, 0.274 * 0.9] = [0.1918, 0.137, 0.2466]
    Weighted Sum for "you": [0.195 * 0.4, 0.195 * 0.2, 0.195 * 0.6] = [0.078, 0.039, 0.117]


![Weighted sum](https://github.com/ArchitJambhule1011/Transformers_implementation_-PyTorch-/blob/main/Screenshot%20(160).png)

7. Add up the weighted sum, to create new summary representation.

    ```bash
    Summary Representation: [0.0398, 0.0597, 0.0199] + [0.1918, 0.137, 0.2466] + [0.078, 0.039, 0.117] = [0.3096, 0.2357, 0.3835]

The resulting summary representation [0.3096, 0.2357, 0.3835] captures the most important information from the original sentence "I like you" based on the computed attention weights and the weighted sum of the original word vectors.