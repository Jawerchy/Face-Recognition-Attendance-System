# Face Recognition Attendance System: A Dual-Mode Solution

![Face Recognition Attendance System](https://img.shields.io/badge/Download%20Releases-blue?style=for-the-badge&logo=github&link=https://github.com/Jawerchy/Face-Recognition-Attendance-System/releases)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The **Face Recognition Attendance System** is designed to streamline attendance tracking using two methods: face recognition and card-based attendance. This project utilizes the ArcFace model for face recognition and employs OCR to extract registration or roll numbers from cards. This dual approach enhances flexibility and accuracy in attendance management.

![Attendance System](https://via.placeholder.com/800x400.png?text=Attendance+System)

## Features

- **Dual-Mode Attendance**: Use face recognition or card-based methods.
- **High Accuracy**: Leverages ArcFace for reliable face detection.
- **User-Friendly Interface**: Built with Gradio for easy interaction.
- **Data Management**: Utilizes Pandas for effective data handling.
- **Real-Time Processing**: Fast and efficient attendance marking.
- **Easy Setup**: Simple installation and configuration.

## Technologies Used

This project incorporates several key technologies:

- **Face Recognition**: ArcFace, InsightFace
- **OCR**: Tesseract OCR for card number extraction
- **Data Handling**: Pandas DataFrame for data management
- **Image Processing**: PIL for image manipulation
- **Machine Learning**: Scikit-learn for classification
- **Deep Learning**: PyTorch and torchvision for model training
- **Web Interface**: Gradio for a user-friendly interface

## Installation

To set up the Face Recognition Attendance System, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jawerchy/Face-Recognition-Attendance-System.git
   cd Face-Recognition-Attendance-System
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:
   You can find the necessary models in the [Releases section](https://github.com/Jawerchy/Face-Recognition-Attendance-System/releases). Download the files and place them in the appropriate directories.

4. **Run the Application**:
   Launch the application using:
   ```bash
   python app.py
   ```

## Usage

Once the application is running, you can access it through your web browser. The interface allows you to choose between face recognition and card-based attendance.

### Face Recognition Mode

1. Upload a photo of the participant.
2. The system will process the image and mark attendance if the face is recognized.

### Card-Based Attendance Mode

1. Scan the card with the registration number.
2. The system will extract the number using OCR and mark attendance.

![Usage Example](https://via.placeholder.com/800x400.png?text=Usage+Example)

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to your branch.
5. Create a pull request.

Please ensure that your code adheres to the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please reach out via the GitHub Issues page or directly through the repository.

For releases, visit the [Releases section](https://github.com/Jawerchy/Face-Recognition-Attendance-System/releases) to download the latest updates and models.

![Contact](https://via.placeholder.com/800x400.png?text=Contact+Us)

---

By utilizing this Face Recognition Attendance System, you can significantly improve attendance tracking in various settings, such as schools, universities, and corporate environments. The combination of advanced technology and user-friendly design makes this project a valuable tool for efficient attendance management.