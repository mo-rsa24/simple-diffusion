## Composing Color and Shape: A Toy Experiment with Diffusion Models
**Objective:** Producing specific combination of colored shapes that generative models have not seen during training

**Motivation** Serves as a proxy to compose co-occurence of diseases on a chest x-ray.

### The Core Idea: Separating and Recombining Generative Forces

The central principle is to train two distinct energy-based or score-based models on a simple dataset. 
- One model will learn the underlying "score" or gradient of the data distribution related to shape
- While the other will learn the score related to color. 

During image generation (inference), we can then guide the diffusion process by simply adding these two scores together. 

This combined score will direct the model to generate an image that satisfies both the shape and color constraints simultaneously.

---

### Experimental Setup

#### **1. The Toy Dataset: Colored Shapes**

To keep things simple and interpretable, we'll create a synthetic dataset of simple geometric shapes with distinct colors.

* **Shapes:** Circles, Squares, and Triangles.
* **Colors:** Red, Green, and Blue.
* **Dataset Generation:** Create a set of images (e.g., 64x64 pixels) showing each shape in each color. For instance, you'll have red circles, green squares, blue triangles, etc. To make the task more interesting and test true compositionality, you can intentionally leave out certain combinations from the training set (e.g., never show a blue circle).

#### **2. Training the Individual Score Models**

We will train two separate conditional diffusion models on this dataset. A diffusion model learns to denoise an image, and in doing so, it implicitly learns the score function ($\nabla_x \log p(x)$), which points in the direction of higher data density.

* **Model Architecture:** A simple U-Net architecture is a standard choice for diffusion models on image data.

* **Shape-Specific Model ($E_{\text{shape}}(x, y_{\text{shape}})$):**
    * **Training:** Train a conditional diffusion model where the condition ($y_{\text{shape}}$) is the shape label (e.g., "circle," "square," "triangle"). The model will learn to generate the correct shape irrespective of its color. To achieve this, during training, for a given shape, you would provide images of that shape in all the colors you have.

* **Color-Specific Model ($E_{\text{color}}(x, y_{\text{color}})$):**
    * **Training:** Train a second conditional diffusion model where the condition ($y_{\text{color}}$) is the color label (e.g., "red," "green," "blue"). This model will learn to generate the overall color characteristics of an image, regardless of the specific shape present. For a given color, you would train on images of all shapes in that color.

#### **3. Composing the Scores at Inference Time**

This is where the magic happens. To generate an image with a specific shape and color, we will run the standard reverse diffusion process, but with a modified score function.

* **The Reverse Diffusion Process:** Starting with pure noise ($x_T$), at each timestep $t$, we will predict the noise to be removed.
* **The Composed Score:** Instead of using the score from a single model, we will combine the scores from our two specialist models:

    $$s_{\text{composed}}(x_t, y_{\text{shape}}, y_{\text{color}}) = s_{\text{shape}}(x_t, y_{\text{shape}}) + s_{\text{color}}(x_t, y_{\text{color}})$$

    Here, $s_{\text{shape}}$ is the score predicted by the shape model and $s_{\text{color}}$ is the score predicted by the color model.

* **Guidance:** In practice, you can also weight the contribution of each model:

    $$s_{\text{composed}} = w_{\text{shape}} \cdot s_{\text{shape}} + w_{\text{color}} \cdot s_{\text{color}}$$

    where $w_{\text{shape}}$ and $w_{\text{color}}$ are scalar weights that control the influence of each attribute.

#### **4. Evaluation: Did it Work?**

The success of the experiment is determined by the model's ability to generate the desired combinations, especially those it has never seen before.

* **Unseen Combinations:** The primary test is to provide the composed model with a shape-color pair that was absent from the training data (e.g., "blue" and "circle" if blue circles were excluded). A successful composition will generate a blue circle.

* **Attribute Control:** Systematically vary the shape and color conditions and observe the output. For a fixed shape condition (e.g., "triangle"), iterate through all color conditions ("red," "green," "blue") and verify that the generated images are all triangles of the specified colors. Do the same for a fixed color and varying shapes.

* **Classifier-Based Evaluation:** For a more quantitative approach, you can train two simple classifiers: one for shape and one for color, on your original dataset. Then, use these classifiers to evaluate the generated images. For a desired output of a "red square," the shape classifier should predict "square" with high confidence, and the color classifier should predict "red" with high confidence.

---

### Why This Works: The Energy-Based Interpretation

This composition is elegantly explained through the lens of Energy-Based Models (EBMs). The probability of an image $x$ is given by $p(x) = \frac{e^{-E(x)}}{Z}$, where $E(x)$ is the energy function and $Z$ is a normalization constant. A lower energy corresponds to a more "likely" image.

When we compose models, we are essentially creating a new energy function that is the sum of the individual energy functions:

$$E_{\text{composed}}(x) = E_{\text{shape}}(x) + E_{\text{color}}(x)$$

An image will have low composed energy only if it has low energy under *both* the shape and color models. Since the score is the gradient of the log-probability (or negative energy), adding the scores is equivalent to combining the underlying energy functions in this way. This provides a principled and powerful way to combine learned knowledge from different generative models.