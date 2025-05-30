import os
import io
import json
import requests # For making HTTP requests to various APIs
import pandas as pd # For handling Excel files
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename # For handling file uploads securely

# --- Configuration (Replace with your actual keys and settings) ---
# It's best to load these from environment variables or a config file
FLASK_SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key')
WHATSAPP_VERIFY_TOKEN = os.environ.get('WHATSAPP_VERIFY_TOKEN', 'your_whatsapp_verify_token') # From Meta App Setup
WHATSAPP_ACCESS_TOKEN = os.environ.get('WHATSAPP_ACCESS_TOKEN', 'your_whatsapp_access_token') # From Meta App
WHATSAPP_PHONE_NUMBER_ID = os.environ.get('WHATSAPP_PHONE_NUMBER_ID', 'your_whatsapp_phone_number_id')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your_openai_api_key')
AIRTABLE_API_KEY = os.environ.get('AIRTABLE_API_KEY', 'your_airtable_api_key')
AIRTABLE_BASE_ID = os.environ.get('AIRTABLE_BASE_ID', 'your_airtable_base_id')
AIRTABLE_TABLE_NAME = os.environ.get('AIRTABLE_TABLE_NAME', 'Transactions') # Example table

# OCR_API_URL = 'YOUR_OCR_API_ENDPOINT' # Example if using a cloud OCR service
# OCR_API_KEY = 'YOUR_OCR_API_KEY'

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = 'uploads' # Create this folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Mock Conversation Memory (In a real app, use a database like Redis or a more robust solution) ---
conversation_memory = {}

# --- Helper Functions ---

def send_whatsapp_message(to_phone_number, message_body):
    """Sends a message back to the user via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_phone_number,
        "type": "text",
        "text": {"body": message_body},
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Message sent to {to_phone_number}: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp message: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def process_image_with_ocr(file_path):
    """
    Placeholder for OCR processing of an image.
    Replace with actual OCR library calls (e.g., Pytesseract, Google Vision AI, AWS Textract).
    """
    print(f"Processing image {file_path} with OCR...")
    # Example with a hypothetical OCR API:
    # with open(file_path, 'rb') as f:
    #     response = requests.post(OCR_API_URL, files={'file': f}, headers={'X-API-Key': OCR_API_KEY})
    #     response.raise_for_status()
    #     return response.json().get('text', '') # Assuming OCR API returns JSON with a 'text' field
    # For local Tesseract (install tesseract-ocr and pytesseract library):
    # try:
    #   import pytesseract
    #   from PIL import Image
    #   text = pytesseract.image_to_string(Image.open(file_path))
    #   return text
    # except ImportError:
    #   print("Pytesseract or Pillow not installed. OCR for images not available.")
    #   return "OCR for image not available. Placeholder text."
    return f"Extracted text from image {os.path.basename(file_path)} (OCR placeholder)"

def process_pdf_with_ocr(file_path):
    """
    Placeholder for OCR processing of a PDF.
    Replace with actual PDF OCR library calls (e.g., PyPDF2 + Pytesseract, cloud OCR).
    """
    print(f"Processing PDF {file_path} with OCR...")
    # Example: use pdf2image to convert PDF pages to images, then OCR each image
    # from pdf2image import convert_from_path
    # try:
    #     images = convert_from_path(file_path)
    #     full_text = ""
    #     for i, image in enumerate(images):
    #         # You might save the image temporarily to pass to process_image_with_ocr
    #         temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_pdf_page_{i}.png")
    #         image.save(temp_image_path, "PNG")
    #         full_text += process_image_with_ocr(temp_image_path) + "\n"
    #         os.remove(temp_image_path) # Clean up
    #     return full_text
    # except Exception as e:
    #     print(f"Error processing PDF with OCR: {e}")
    #     return "OCR for PDF not available. Placeholder text."
    return f"Extracted text from PDF {os.path.basename(file_path)} (OCR placeholder)"

def parse_data_with_llm(text_content, user_phone_number, is_query=False):
    """
    Uses OpenAI's GPT to parse text into structured accounting data or answer queries.
    """
    print(f"Parsing text with LLM: {text_content[:100]}...")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    user_history = conversation_memory.get(user_phone_number, [])
    messages = [{"role": "system", "content": "You are an intelligent accounting assistant. If the input is financial data, extract details like vendor, item description, quantity, unit price, total amount, transaction date, and category (e.g., fuel, office supplies, revenue). Respond in JSON format. If it's a question about finances, use the provided data (if any) or general knowledge to answer."}]

    # Add conversation history
    for entry in user_history[-5:]: # last 5 turns
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["ai"]})

    messages.append({"role": "user", "content": text_content})

    if is_query:
        prompt_instruction = (
            "The user is asking a question. Based on their question and our conversation history, "
            "identify what financial data they need. Formulate a response. If you need to query a database, "
            "suggest the query parameters (e.g., date range, category). For now, just answer based on the context provided or state you need to query a database."
            # In a real app, you might have the LLM generate Airtable query filters.
        )
    else: # Data extraction
        prompt_instruction = (
            "The user provided some financial data. Extract the following details: "
            "vendor_name, transaction_date (YYYY-MM-DD), line_items (array of objects with description, quantity, unit_price, total_amount), "
            "grand_total, currency, expense_category (e.g., Fuel, Maintenance, Office Supplies, Travel, Salary, Rent, Utilities, Software, Marketing), "
            "or income_category (e.g., Sales, Services, Interest). "
            "If some fields are missing, set them to null. If it's an invoice, it could be income or expense. "
            "If it's a receipt, it's likely an expense. Use context. Respond ONLY with a JSON object."
        )
    messages[-1]["content"] = f"{prompt_instruction}\n\nUser input: {text_content}"


    data = {
        "model": "gpt-3.5-turbo", # Or gpt-4 for more complex tasks
        "messages": messages,
        "response_format": {"type": "json_object"} if not is_query else None # Use JSON mode for structured data
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        llm_response_data = response.json()
        llm_output = llm_response_data['choices'][0]['message']['content']
        print(f"LLM Output: {llm_output}")
        if not is_query:
            return json.loads(llm_output) # Expecting JSON for data extraction
        return llm_output # Text for queries
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        if e.response is not None:
            print(f"OpenAI Error Response: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM: {e}. Response was: {llm_output}")
        return None


def save_to_airtable(structured_data, user_phone_number):
    """Saves structured data to Airtable."""
    print(f"Saving to Airtable: {structured_data}")
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        print("Airtable credentials not set. Skipping save.")
        return None

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

    # --- Adapt this mapping to your Airtable schema ---
    # This is a very simplified example. You'll need to handle line items, etc.
    fields = {
        "UserPhone": user_phone_number,
        "Vendor": structured_data.get("vendor_name"),
        "Date": structured_data.get("transaction_date"),
        "Description": ", ".join([item.get('description', '') for item in structured_data.get("line_items", [])]) if structured_data.get("line_items") else structured_data.get("description"),
        "TotalAmount": structured_data.get("grand_total"),
        "Category": structured_data.get("expense_category") or structured_data.get("income_category"),
        "RawData": json.dumps(structured_data) # Store the full LLM output for reference
    }
    # Filter out None values to avoid Airtable errors for empty optional fields
    fields = {k: v for k, v in fields.items() if v is not None}


    data = {"records": [{"fields": fields}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Data saved to Airtable: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error saving to Airtable: {e}")
        if e.response is not None:
            print(f"Airtable Error Response: {e.response.text}")
        return None

def query_airtable_for_llm(query_params):
    """
    Queries Airtable based on parameters derived by LLM.
    This is highly conceptual. LLM would need to output structured query filters.
    Example query_params: {"category": "Fuel", "date_range": {"start": "2024-04-01", "end": "2024-04-30"}}
    """
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
        print("Airtable credentials not set. Skipping query.")
        return "Could not query Airtable: credentials missing."

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    
    # --- Build Airtable filterByFormula string ---
    # This is complex and needs careful construction based on LLM output
    # For example, if LLM gives: {"Category": "Fuel", "DateMin": "2023-01-01", "DateMax": "2023-01-31"}
    # Formula: AND({Category} = 'Fuel', IS_AFTER({Date}, '2023-01-01'), IS_BEFORE({Date}, '2023-01-31'))
    filter_parts = []
    if query_params.get("Category"):
        filter_parts.append(f"({{Category}} = '{query_params['Category']}')")
    if query_params.get("DateMin"):
         filter_parts.append(f"(IS_AFTER({{Date}}, '{query_params['DateMin']}'))") # Or IS_SAME or include date itself
    if query_params.get("DateMax"):
         filter_parts.append(f"(IS_BEFORE({{Date}}, '{query_params['DateMax']}'))")

    filter_formula = ""
    if filter_parts:
        filter_formula = f"AND({', '.join(filter_parts)})"
        url += f"?filterByFormula={requests.utils.quote(filter_formula)}"

    print(f"Querying Airtable with URL: {url}")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        records = response.json().get("records", [])
        # Process records to a summary string for the LLM or user
        if not records:
            return "No matching records found."

        summary_lines = []
        total_amount = 0
        for record in records:
            fields = record.get("fields", {})
            desc = fields.get("Description", "N/A")
            amount = fields.get("TotalAmount", 0)
            date = fields.get("Date", "N/A")
            summary_lines.append(f"- {date}: {desc} (${amount})")
            if isinstance(amount, (int, float)):
                total_amount += amount
        
        summary_result = f"Found {len(records)} records. Total: ${total_amount:.2f}\n" + "\n".join(summary_lines)
        return summary_result

    except requests.exceptions.RequestException as e:
        print(f"Error querying Airtable: {e}")
        if e.response is not None: print(f"Airtable Error: {e.response.text}")
        return "Sorry, I encountered an error trying to fetch your data from Airtable."


def update_conversation_memory(user_phone, user_input, ai_response):
    if user_phone not in conversation_memory:
        conversation_memory[user_phone] = []
    conversation_memory[user_phone].append({"user": user_input, "ai": ai_response})
    # Keep memory from getting too large
    if len(conversation_memory[user_phone]) > 10:
        conversation_memory[user_phone].pop(0)


# --- Flask Webhook Routes ---

@app.route('/webhook', methods=['GET', 'POST'])
def whatsapp_webhook():
    if request.method == 'GET':
        # WhatsApp Business API Verification
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if mode == 'subscribe' and token == WHATSAPP_VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            print("VERIFICATION_FAILED")
            return 'Verification failed', 403

    elif request.method == 'POST':
        data = request.get_json()
        print(f"Received WhatsApp message: {json.dumps(data, indent=2)}")

        if data.get('object') == 'whatsapp_business_account':
            for entry in data.get('entry', []):
                for change in entry.get('changes', []):
                    if change.get('field') == 'messages':
                        value = change.get('value', {})
                        metadata = value.get('metadata', {})
                        messages = value.get('messages', [])

                        if not messages:
                            continue
                        
                        message_obj = messages[0]
                        from_phone = message_obj.get('from')
                        message_type = message_obj.get('type')
                        
                        response_message = "Sorry, I didn't understand that." # Default
                        llm_query_response = None

                        if message_type == 'text':
                            text_content = message_obj['text']['body']
                            print(f"Processing text message from {from_phone}: {text_content}")
                            # Determine if it's a query or data submission
                            # Simple heuristic: if it contains typical question words.
                            # A more robust way is to ask LLM to classify intent.
                            is_question = any(word in text_content.lower() for word in ["how much", "what is", "total", "list", "show me"])
                            
                            if is_question:
                                # Conceptual: LLM first helps understand what to query from Airtable
                                # This might be a multi-step process in a real app
                                # 1. LLM to parse question into query parameters
                                # For now, we'll just assume the text_content is the query
                                print(f"Interpreting as a query: {text_content}")
                                # This part needs more sophisticated LLM prompting to extract query params for Airtable
                                # For simplicity, let's say our LLM could directly generate Airtable filter logic or a summary
                                # Example: if LLM says "user wants fuel expenses for last month", then call query_airtable_for_llm
                                # This requires a sophisticated LLM prompt and parsing.
                                # Let's assume a very basic query structure for now.
                                query_params_from_llm = {} # This should be output of an LLM call to interpret text_content
                                if "fuel" in text_content.lower() and "last month" in text_content.lower():
                                    query_params_from_llm = {"Category": "Fuel", "DateMin": "2024-04-01", "DateMax": "2024-04-30"} # Example
                                
                                if query_params_from_llm:
                                    airtable_results = query_airtable_for_llm(query_params_from_llm)
                                    # Then, potentially pass this to LLM again for a natural language summary
                                    llm_query_response = parse_data_with_llm(f"Summarize this data for the user: {airtable_results}", from_phone, is_query=True)
                                else: # General query to LLM
                                    llm_query_response = parse_data_with_llm(text_content, from_phone, is_query=True)

                                response_message = llm_query_response or "I couldn't process your query at the moment."

                            else: # Data submission
                                structured_data = parse_data_with_llm(text_content, from_phone, is_query=False)
                                if structured_data:
                                    save_to_airtable(structured_data, from_phone)
                                    response_message = "Thanks! I've processed your text data and saved it."
                                else:
                                    response_message = "I received your text, but I couldn't structure it. Can you try rephrasing?"
                        
                        elif message_type in ['image', 'document']: # Document could be PDF or Excel
                            media_id = message_obj.get(message_type, {}).get('id')
                            mime_type = message_obj.get(message_type, {}).get('mime_type', '')
                            filename = message_obj.get(message_type, {}).get('filename', 'downloaded_file') # For documents

                            if not media_id:
                                response_message = "Error: Media ID not found."
                            else:
                                # 1. Get Media URL from Meta
                                media_url_ep = f"https://graph.facebook.com/v19.0/{media_id}"
                                headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
                                media_info_resp = requests.get(media_url_ep, headers=headers)
                                
                                if media_info_resp.status_code == 200:
                                    media_url = media_info_resp.json().get('url')
                                    # 2. Download the media
                                    media_content_resp = requests.get(media_url, headers=headers)
                                    if media_content_resp.status_code == 200:
                                        # Ensure filename is secure and has an extension
                                        if '.' not in filename and mime_type:
                                            ext = mime_type.split('/')[-1]
                                            if ext == 'plain': ext = 'txt' # for text/plain
                                            filename = f"{secure_filename(filename)}.{ext}"
                                        else:
                                            filename = secure_filename(filename)

                                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                        with open(file_path, 'wb') as f:
                                            f.write(media_content_resp.content)
                                        print(f"File {filename} downloaded to {file_path}")

                                        extracted_text = None
                                        if message_type == 'image' or mime_type == 'image/jpeg' or mime_type == 'image/png':
                                            extracted_text = process_image_with_ocr(file_path)
                                        elif mime_type == 'application/pdf':
                                            extracted_text = process_pdf_with_ocr(file_path)
                                        elif 'excel' in mime_type or 'spreadsheetml' in mime_type:
                                            try:
                                                df = pd.read_excel(file_path)
                                                # Convert relevant parts of Excel to text for LLM
                                                # This might need more sophisticated handling based on Excel structure
                                                extracted_text = "Excel data: \n" + df.to_string(index=False, header=True)
                                                print(f"Read Excel data: {extracted_text[:200]}")
                                            except Exception as e:
                                                print(f"Error reading Excel: {e}")
                                                extracted_text = "Could not process the Excel file."
                                        else:
                                            response_message = f"Unsupported document type: {mime_type}"
                                        
                                        if extracted_text:
                                            structured_data = parse_data_with_llm(extracted_text, from_phone, is_query=False)
                                            if structured_data:
                                                save_to_airtable(structured_data, from_phone)
                                                response_message = f"Processed your {message_type} and saved the data!"
                                            else:
                                                response_message = f"I got the {message_type}, but couldn't structure the data from it."
                                        
                                        # Clean up uploaded file
                                        # os.remove(file_path) 
                                        # Consider if you want to keep it for debugging or archive
                                    else:
                                        response_message = "Failed to download the media."
                                else:
                                    response_message = "Failed to get media URL."
                        else:
                            response_message = f"Sorry, I can only process text, images, PDFs, and Excel files for now. Received: {message_type}"

                        send_whatsapp_message(from_phone, response_message)
                        update_conversation_memory(from_phone, message_obj.get(message_type, {}).get('body', f'[{message_type.upper()}]'), response_message)
            return jsonify({'status': 'ok'}), 200
        else:
            # Not a WhatsApp business account update
            return 'Not a WhatsApp message', 404


if __name__ == '__main__':
    # Make sure to configure SSL if exposing this publicly for WhatsApp Webhooks
    print("Starting Flask app...")
<<<<<<< HEAD
    app.run(debug=True, port=5000) # Default port 5000, use 0.0.0.0 for host in docker
=======
    app.run(debug=True, port=5000) # Default port 5000, use 0.0.0.0 for host in docker
>>>>>>> 322b1ad (Update index.py)
