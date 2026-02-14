import time
import schedule

from framework.trading.bot import TradingBot


def start_bot() -> None:
    try:
        bot = TradingBot()

        # Schedule
        schedule.every(1).minutes.do(bot.run_cycle)

        # Run once immediately
        bot.run_cycle()

        print("Bot is running logic loop...")
        while True:
            schedule.run_pending()
            time.sleep(1)

    except Exception as e:
        print(f"Fatal Bot Error: {e}")


if __name__ == "__main__":
    start_bot()
