{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a4a55022",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Weights: -0.27210172434845314, -0.21061203639789283, -0.4202080230763725, -0.2672091136389698, -0.3989985705902709, -0.22202639688990788\n",
      "Biases: 0.5, 0.7\n",
      "Input: 0.1, 0.2\n",
      "Output: 0.4242271560142135\n"
     ]
    }
   ],
   "source": [
    "import random\n",
    "import math\n",
    "\n",
    "def tanh(x):\n",
    "    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))\n",
    "\n",
    "class SimpleNN:\n",
    "    def __init__(self):\n",
    "        self.w1 = random.uniform(-0.5, 0.5)\n",
    "        self.w2 = random.uniform(-0.5, 0.5)\n",
    "        self.w3 = random.uniform(-0.5, 0.5)\n",
    "        self.w4 = random.uniform(-0.5, 0.5)\n",
    "        self.w5 = random.uniform(-0.5, 0.5)\n",
    "        self.w6 = random.uniform(-0.5, 0.5)\n",
    "        self.b1 = 0.5\n",
    "        self.b2 = 0.7\n",
    "\n",
    "    def forward(self, x1, x2):\n",
    "        h1 = tanh(self.w1 * x1 + self.w2 * x2 + self.b1)\n",
    "        h2 = tanh(self.w3 * x1 + self.w4 * x2 + self.b1)\n",
    "        output = tanh(self.w5 * h1 + self.w6 * h2 + self.b2)\n",
    "        return output\n",
    "\n",
    "nn = SimpleNN()\n",
    "x1, x2 = 0.1, 0.2\n",
    "output = nn.forward(x1, x2)\n",
    "\n",
    "print(f\"Weights: {nn.w1}, {nn.w2}, {nn.w3}, {nn.w4}, {nn.w5}, {nn.w6}\")\n",
    "print(f\"Biases: {nn.b1}, {nn.b2}\")\n",
    "print(f\"Input: {x1}, {x2}\")\n",
    "print(f\"Output: {output}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
