
from AkanechanBot import AkanechanBot

bot = AkanechanBot()

while True :
    inputMessage = input("あなた > ").strip()
    replay = bot.replay(inputMessage)
    print("\t"*4, f"{replay} < アカネチャン")
