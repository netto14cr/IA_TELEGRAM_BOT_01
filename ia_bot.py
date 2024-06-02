import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

class TelegramChatBot:
    def __init__(self, telegram_token):
        self.telegram_token = telegram_token
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.application = Application.builder().token(telegram_token).build()
        self.application.add_handler(CommandHandler('start', self.start))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.awaiting_question = False
        self.awaiting_context = False
        self.awaiting_another_question = False
        self.question = ""
        self.context = ""

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.awaiting_question:
            self.question = update.message.text
            await update.message.reply_text('Please provide the context.')
            self.awaiting_question = False
            self.awaiting_context = True
        elif self.awaiting_context:
            self.context = update.message.text
            answer = self.query(self.question, self.context)
            await update.message.reply_text(answer)
            await self.ask_another_question(update)
            self.awaiting_context = False
            self.awaiting_another_question = True
        elif self.awaiting_another_question:
            if update.message.text.lower() == 'yes':
                await update.message.reply_text('Please ask a question.')
                self.awaiting_question = True
                self.awaiting_another_question = False
            elif update.message.text.lower() == 'no':
                await update.message.reply_text('Goodbye!')
                self.awaiting_another_question = False
            else:
                await update.message.reply_text('Please respond with "Yes" or "No".')
        else:
            await update.message.reply_text('Please ask a question first.')

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text('Hello! I am a question-answering bot. Please ask me a question.')
        self.awaiting_question = True

    def query(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        outputs = self.model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        return answer

    async def ask_another_question(self, update: Update):
        buttons = [[KeyboardButton("Yes"), KeyboardButton("No")]]
        keyboard_markup = ReplyKeyboardMarkup(buttons, one_time_keyboard=True, resize_keyboard=True)
        try:
            await update.message.reply_text("Do you want to ask another question?", reply_markup=keyboard_markup)
        except Exception as e:
            print(f"An error occurred while sending the message: {e}")

    def run(self):
        self.application.run_polling()

if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    bot = TelegramChatBot(TELEGRAM_TOKEN)
    bot.run()
