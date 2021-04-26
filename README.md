# <span>Bengali.AI</span> Handwritten Grapheme Classification Bronze Medal Solution

<!-- ABOUT THE PROJECT -->
## About The Project

<br/>
<p align="center">
  <img src="/image/image.png" alt="Competition image" width="500" height="500"/>
</p>


<!-- ![Product Name Screen Shot](/image/image.png) -->

This is my solution to the [Bengali.AI Handwritten Grapheme Classification competition](https://www.kaggle.com/c/bengaliai-cv19/overview). I used a DenseNet-121 that receives a one-channel image and outputs 3 separate classifiers for the grapheme root, vowel diacritics, and consonant diacritics, respectively. The MixUp augmentation is used to improve the performance.

Result: Macro-average recall score of 0.9319 in the [private leaderboard](https://www.kaggle.com/c/bengaliai-cv19/leaderboard). Ranked 182 out of 2059 teams (bronze medal region).
<br/><br/>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple example steps.
<br/><br/>

### Prerequisites

* PyTorch (version 1.3.0)

  Install using Anaconda:
  ```sh
  conda install pytorch=1.3.0 -c pytorch
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/stephenllh/bengali_character.git
   ```

1. Change directory
   ```sh
   cd bengali_character
   ```

2. Install packages
   ```sh
   pip install requirements.txt
   ```
<br/>

<!-- USAGE EXAMPLES -->
## Usage

1. Change directory
   ```sh
   cd bengali_character
   ```

2. Download the dataset
    - Option 1: Use Kaggle API
      - `pip install kaggle`
      - `kaggle competitions download -c bengaliai-cv19`
    - Option 2: Download the dataset from the [competition website](https://github.com/).

3. Run the training script
   ```sh
   python train.py
   ```

4. (Optional) Run the inference script
   ```sh
   python inference.py
   ```

<br/>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
<br></br>


<!-- CONTACT -->
## Contact

Stephen Lau - [Email](stephenlaulh@gmail.com) - [Twitter](https://twitter.com/StephenLLH) - [Kaggle](https://www.kaggle.com/faraksuli)

Project Link: [https://github.com/stephenllh/bengali_handwriting](https://github.com/your_username/repo_name)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
