<div align="center">
    <h1>
    Robots as Furniture
    </h1>
    <p>
    <b>Robots as Furniture</b> is a Python project that enables robots to function as responsive, voice-controlled furniture by interpreting spoken commands through multi-microphone audio processing and OpenAI's AI models, and executing movements via sound source localization and ROS2. <br>
    </p>
    <p>
    <img src="docs/logo.png" alt="Robots as Furniture Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/github/stars/robotsasfurniture/passive-sound-localization
    " alt="Github Stars"></a>
</div>

# Table of Contents
1. [Installation](#installation)
2. [Running the project](#running-the-project)
3. [Acknowledgements](#acknowledgments)
4. [Citation](#citation)

# Installation
As a pre-requisite, you must have [Poetry](https://python-poetry.org/) installed.

In addition, the project uses the [OpenAI API](https://platform.openai.com/docs/overview). In order to use the OpenAI API, you must sign up with OpenAI and get an API key. The API key should be stored securely in an environment variable, such as an `.env` file:

```bash
OPENAI_API_KEY="your_api_key_here"
```


To install the project, clone the git repo by running the following commands in your terminal:
```bash
git clone https://github.com/robotsasfurniture/passive-sound-localization.git
cd passive-sound-localization
```

# Running the project
To run the project, run this bash command in your terminal:
```bash
bash scripts/run.sh
```

# Acknowledgements
We thank the contributors to take the time and energy to contribute to this repo. 

# Citation
If you'd like to cite this project, please use this BibTex:
```
@article{perez2024robotsasfurniture,
  title={Robots as Furniture},
  author={Nicolas Perez, John Finberg, Dave Song and others},
  journal={https://example.com/},
  year={2024}
}
```