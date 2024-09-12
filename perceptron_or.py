import numpy as np
from perceptron import Perceptron

#construct the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

print("[INFO] Training perceptron...")
p = Perceptron(X.shape[1])
p.fit(X, y, epochs=20)

print("[INFO] Testing perceptron...")

z = input("[INFO] Do you want to test perceptron your own? y/n")

while(z.lower() == "y"):
    n1 = int(input("Give n1 (0 or 1):"))
    n2 = int(input("Give n2 (0 or 1):"))

    test_input = np.array([[n1, n2]])
    prediction = p.predict(test_input)
    print(f"[INFO] Prediction for ({n1}, {n2}) is: {prediction}")

    z = input("[INFO] Do you want to continue testing?")