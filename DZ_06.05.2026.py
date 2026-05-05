import random

secret_word = "king"

print("Contexto")

while True:
    guess = input("Enter your guess: ").strip().lower()

    if guess not in model.wv:
        print("that word is not in the dictionary")
        continue

    if guess == secret_word:
        print("you win!")
        break

    similarity = model.wv.similarity(secret_word, guess)
    
    if similarity > 0.7:
        print(f"close! similarity: {similarity:.4f}")
    else:
        print(f"far away. similarity: {similarity:.4f}")
