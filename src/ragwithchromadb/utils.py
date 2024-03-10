import os
import PyPDF2
import json
import traceback
import re


def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        except Exception as e:
            raise Exception("error reading the PDF file")

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise Exception(
            "unsupported file format only pdf and text file suppoted"
        )


def convert_pdf_to_text(pdf_file):
    """Converts a PDF file to text and saves it as a .txt file.

    Args:
        pdf_file: The file object representing the PDF file.

    Raises:
         Exception: If the file is not a PDF or an error occurs during processing.
    """

    if not pdf_file.name.endswith(".pdf"):
        raise Exception("Only PDF files are supported")

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Construct the output filename (replace .pdf with .txt)
        output_filename = pdf_file.name[:-4] + ".txt"

        with open(output_filename, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        return output_filename

    except Exception as e:
        raise Exception("Error processing the PDF file") from e



def get_table_data(quiz_str):
    try:
        print("Debug: quiz_str: " + quiz_str)

        # Attempt to extract valid JSON within the string
        json_pattern = r"\{.*\}"  # Basic JSON pattern
        json_match = re.search(json_pattern, quiz_str)

        if json_match:
            clean_json_str = json_match.group(0)

            # Now work with the clean JSON string
            quiz_dict = json.loads(clean_json_str)
            quiz_table_data = []

            # iterate over the quiz dictionary and extract the required information
            for key, value in quiz_dict.items():
                mcq = value["mcq"]
                options = "\n".join(
                    [
                        f" {option}. {option_value}, " for option, option_value in value["options"].items()
                    ]
                )
                correct = value["correct"]
                quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

            return quiz_table_data
        else:
            print("Error: No valid JSON structure found in quiz_str")
            return False

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
