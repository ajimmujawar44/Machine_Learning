import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Simple Tkinter App")
root.geometry("300x150")


# Function to update the label
def say_hello():
    message.config(text="Hello, World!")


# Label
message = tk.Label(root, text="Click the button")
message.pack(pady=10)

# Button
hello_button = tk.Button(
    root,
    text="Click Me",
    command=say_hello
)
hello_button.pack()

# Run the application
root.mainloop()