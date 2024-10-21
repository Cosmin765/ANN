import pyautogui
import random
import time
import math

pyautogui.FAILSAFE = False


def random_mouse_movement(duration=60):
    screen_width, screen_height = pyautogui.size()
    start_time = time.time()
    # acc = 600

    while True:
        # st = time.time()
        # Randomly choose a new target location on the screen
        target_x = random.randint(0, screen_width - 1)
        target_y = random.randint(0, screen_height - 1)

        # Get the current mouse position
        current_x, current_y = pyautogui.position()

        # Calculate the distance and angle for smooth movement
        distance = math.hypot(target_x - current_x, target_y - current_y)
        steps = int(distance / 5)  # Determine the number of steps based on the distance
        steps = max(steps, 10)  # Ensure there are enough steps for smooth movement

        # Interpolate the movement path with slight randomness
        for i in range(steps):
            intermediate_x = current_x + (target_x - current_x) * (i / steps)
            intermediate_y = current_y + (target_y - current_y) * (i / steps)

            # Add a slight random deviation to simulate human-like movement
            jitter_x = random.uniform(-1, 1)
            jitter_y = random.uniform(-1, 1)

            pyautogui.moveTo(intermediate_x + jitter_x, intermediate_y + jitter_y, duration=random.uniform(0.01, 0.03))

        # Random pause to simulate human hesitation
        time.sleep(random.uniform(0.5, 2.0))
        #amount = time.time() - st
        #acc -= amount
        #if acc <= 0:
        #    acc = 600
        #    pyautogui.hotkey('ctrl', 'r')

if __name__ == "__main__":
    random_mouse_movement(duration=60)  # Run for 60 seconds
