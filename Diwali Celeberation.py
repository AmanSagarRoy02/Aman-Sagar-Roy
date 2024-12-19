import turtle
import random
import time
import pygame

# Initialize pygame mixer for sound
pygame.init()
pygame.mixer.init()

# Load the firecracker sound file
try:
    pygame.mixer.music.load("C:/Users/amans/Downloads/fireworks-close-29630.mp3")
except pygame.error as e:
    print(f"Error loading sound file: {e}")

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Happy Diwali From A.S.R!")
screen.setup(width=800, height=600)

# Create turtle objects for fireworks and text
firework = turtle.Turtle()
firework.hideturtle()
firework.speed(0)
firework.width(2)

text_turtle = turtle.Turtle()
text_turtle.hideturtle()
text_turtle.speed(0)

# List of colors for fireworks and text glow
colors = ["red", "orange", "yellow", "green", "blue", "purple", "white", "pink", "cyan", "magenta"]

# Function to create a single firework at a random position
def create_firework():
    if screen._root.winfo_exists():  # Check if the Turtle screen is still open
        x = random.randint(-300, 300)
        y = random.randint(-200, 200)
        firework.penup()
        firework.goto(x, y)
        firework.pendown()

        firework.color(random.choice(colors))
        size = random.randint(30, 70)

        # Draw explosion effect
        for _ in range(36):
            firework.forward(size)
            firework.backward(size)
            firework.right(10)

        # Play the firecracker sound
        if not pygame.mixer.music.get_busy():  # Check if sound is not already playing
            pygame.mixer.music.play()

# Function to display fireworks continuously
def diwali_fireworks(start_time):
    current_time = time.time()
    if current_time - start_time < 15:  # Run for 15 seconds
        create_firework()  # Create a firework
        screen.ontimer(lambda: diwali_fireworks(start_time), 300)  # Schedule the next firework

# Function for glowing "Happy Diwali From A.S.R!" text effect
def glowing_text(start_time):
    current_time = time.time()
    if current_time - start_time < 15:  # Run for 15 seconds
        text_turtle.clear()
        text_turtle.color(random.choice(colors))
        text_turtle.write("Happy Diwali From A.S.R!", align="center", font=("Arial", 36, "bold"))
        screen.ontimer(lambda: glowing_text(start_time), 400)  # Schedule the next glowing text update

# Function to terminate the program after 15 seconds
def terminate_program():
    turtle.bye()  # Close the Turtle graphics window
    pygame.mixer.music.stop()  # Stop the sound
    print("Program terminated after 15 seconds.")

# Run the animations and sounds
if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    diwali_fireworks(start_time)  # Start fireworks
    glowing_text(start_time)      # Start glowing text
    screen.ontimer(terminate_program, 15000)  # Schedule termination after 15 seconds
    turtle.mainloop()  # Keep the window open until manually closed
