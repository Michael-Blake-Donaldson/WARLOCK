from chatbot import Chatbot

def chat():
    bot = Chatbot()
    print("- WARLOCK ONLINE -")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("WARLOCK out.")
            break
        response = bot.get_response(user_input)
        print(f"WARLOCK: {response}")

        # Learn from the user input
        feedback = input("Was the response appropriate? (yes/no): ").strip().lower()
        if feedback == 'no':
            correct_response = input("Please provide the correct response: ").strip()
            bot.add_training_data((user_input, correct_response))

if __name__ == "__main__":
    chat()
    input("Press Enter to close...")  # This will prevent the terminal from closing immediately