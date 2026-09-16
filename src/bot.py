import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart

from rag_engine import FamilyCodeRAGEngine

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found! Please check your .env file.")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

print("Initializing RAG Engine... (Connecting to database)")
rag = FamilyCodeRAGEngine()
print("System is ready! Bot is waiting for users...")

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """
    Handles the /start command.
    Sends a welcome message explaining the bot's purpose and scope.
    """

    welcome_text = (
        "⚖️ Salam! Mən Azərbaycan Ailə Məcəlləsi üzrə süni intellekt köməkçisiyəm.\n\n"
        "Mən qanunvericilik bazasına (e-qanun.az) əsaslanaraq suallarınızı cavablandırıram.\n"
        "Sualınızı mətn formatında yazaraq mənə göndərə bilərsiniz."
    )
    await message.answer(welcome_text)

@dp.message(F.text)
async def handle_text_messages(message: types.Message):
    """
    Handles incoming text messages from users.
    Queries the RAG engine asynchronously and updates the wait message with the final answer.
    """

    user_question = message.text

    if not user_question.strip():
        await message.answer("Zəhmət olmasa sualınızı yazın. Boş mesaj göndərmisiniz. ✍️")
        return

    wait_message = await message.answer("🔍 Qanunvericilik bazasında axtarış aparılır, zəhmət olmasa gözləyin...")
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        answer = await asyncio.to_thread(rag.answer_question, user_question)
        await wait_message.edit_text(answer)

    except Exception as e:
        print(f"Xəta baş verdi: {e}")
        await wait_message.edit_text("⚠️ Texniki nasazlıq yarandı. Zəhmət olmasa daha sonra yenidən cəhd edin.")

@dp.message(~F.text)
async def handle_non_text_messages(message: types.Message):
    """
    Handles non-text messages (e.g., photos, voice memos, documents).
    Instructs the user to send text-only queries to prevent processing errors.
    """
    await message.answer(
        "Üzr istəyirəm, mən hələlik səsli mesajları, şəkilləri və ya sənədləri oxuya bilmirəm. 🤖\n"
        "Zəhmət olmasa sualınızı yalnız mətn şəklində yazın."
    )

async def main():
    await bot.delete_webhook(drop_pending_updates=True) 
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())