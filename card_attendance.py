import pytesseract
from PIL import Image
import re
import attendance

def extract_registration_number(text):
    """
    Extract registration/roll number from OCR text using regex.
    Adjust the regex pattern to match your registration number format.
    """
    # Example: F23BSCS001 or similar
    match = re.search(r'F\d{2}[A-Z]+\d{3}', text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None

def mark_attendance_from_card(image):
    """
    Extracts registration number from card image and marks attendance.
    Args:
        image (PIL.Image.Image): The card image.
    Returns:
        str: Attendance status message.
    """
    # OCR to extract text
    text = pytesseract.image_to_string(image)
    reg_no = extract_registration_number(text)
    if reg_no:
        message = attendance.record_attendance(reg_no)
        return f"Reg No: {reg_no} - {message}"
    else:
        return "Registration number not found on card. Please try again." 