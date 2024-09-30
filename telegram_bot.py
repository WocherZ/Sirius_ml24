import os
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from retrieval_qa import QASystem
from dotenv import load_dotenv
load_dotenv()

# Словарь для хранения контекста пользователей
user_context = {}

qa_system = QASystem()

# Функция начала /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот, который поможет тебе с ответами на вопросы. Для начала укажи контекст.")


# Функция обработки сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id not in user_context:
        user_context[user_id] = {'context': update.message.text}
        qa_system.init_retriever([user_context[user_id]['context']])
        qa_system.create_qa_chain()
        await update.message.reply_text(f"Контекст установлен. Теперь задай мне вопрос.")
        return

    question = update.message.text

    answer = qa_system.get_answer_by_context(question)

    reply_markup = ReplyKeyboardMarkup([[KeyboardButton("Сменить контекст")]], one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(answer, reply_markup=reply_markup)


# Функция смены контекста
async def change_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user_context.pop(user_id, None)
    await update.message.reply_text("Контекст сброшен. Напиши новый контекст.")


# Основная функция запуска бота
def main() -> None:
    # Здесь вставьте токен вашего бота
    bot_token = os.getenv('BOT_TOKEN')

    # Создание экземпляра приложения
    application = ApplicationBuilder().token(bot_token).build()

    # Команда /start
    application.add_handler(CommandHandler("start", start))

    # Обработка смены контекста
    application.add_handler(MessageHandler(filters.Regex('Сменить контекст'), change_context))

    # Обработка сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    application.run_polling()


if __name__ == "__main__":
    main()
