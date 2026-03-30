import pandas as pd
import matplotlib.pyplot as plt

csvfile = input("Which file? ")
df = pd.read_csv(f"training_output/{csvfile}.csv")

plt.plot(df["epoch"], df[csvfile])

plt.xlim(left=0)
if csvfile == "loss":
    plt.yscale("log") #useful for loss

elif csvfile == "weights":
    plt.xlim(right=1000) #useful for weights

else: 
    pass

plt.xlabel("Epoch")
plt.ylabel(csvfile)
plt.title(f"Training {csvfile}")
plt.show()