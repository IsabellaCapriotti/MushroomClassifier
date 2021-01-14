import pandas as pd
import numpy as np 
import random

##########################################################
# MODEL                                                  #
##########################################################
# Read data
mushroom_data = pd.read_csv('agaricus-lepiota.data', names=['poison-label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
mushroom_data = mushroom_data.reindex(np.random.permutation(mushroom_data.index))

# Separate out labels
mushroom_labels = pd.DataFrame(mushroom_data['poison-label'])
mushroom_data.drop('poison-label', axis=1, inplace=True)


# Convert non-numerical data to numbers
def convert_to_numbers(column, keyToUse=None):
    vals = {}
    currMapIndex = 0

    for item in column: 
        if item not in vals:
            vals[item] = currMapIndex
            currMapIndex += 1
    
    if keyToUse is None:
        column.replace(vals, inplace=True)
    else:
        column.replace(keyToUse, inplace=True)

    return vals

numberMappings = {}
for colName, colData in mushroom_data.iteritems():
    numberMappings[colName] = convert_to_numbers(colData)


# Add cross features
def add_cross_features(mushroom_data):
    mushroom_data['stalk-shape-and-root'] = mushroom_data['stalk-shape'] * mushroom_data['stalk-root']
    mushroom_data['veil-type-and-color'] = mushroom_data['veil-type'] * mushroom_data['veil-color']
    mushroom_data['all-cap'] = mushroom_data['cap-shape'] * mushroom_data['cap-surface'] * mushroom_data['cap-color']
    mushroom_data['all-gill'] =  mushroom_data['gill-size'] * mushroom_data['gill-color'] * mushroom_data['gill-spacing']
    mushroom_data['ring-type-and-number'] = mushroom_data['ring-type'] * mushroom_data['ring-number']
    mushroom_data['stalk-and-cap'] = mushroom_data['stalk-shape-and-root'] * mushroom_data['all-cap']
    mushroom_data['gill-and-veil'] = mushroom_data['all-gill'] * mushroom_data['veil-type-and-color']
    mushroom_data['stalk-and-gill'] = mushroom_data['stalk-shape-and-root'] * mushroom_data['all-gill']

add_cross_features(mushroom_data)

# Convert edible/poisonous to boolean 
mushroom_labels.replace({'p': 0, 'e': 1}, inplace=True)

# Separate into train and test sets
mushroom_data_train = mushroom_data.iloc[:7325]
mushroom_data_test = mushroom_data.iloc[7325:]

mushroom_labels_train = mushroom_labels[:7325]
mushroom_labels_test = mushroom_labels[7325:]

# Convert dataframes to numpy arrays
mushroom_data_train = mushroom_data_train.to_numpy().T
mushroom_data_test = mushroom_data_test.to_numpy().T

mushroom_labels_train = mushroom_labels_train.to_numpy().T
mushroom_labels_test = mushroom_labels_test.to_numpy().T


# Initialize parameters
def initialize_parameters(input_shape):
    W = np.random.randn(input_shape[0],1) * .01
    b = np.random.randn() * .01
    
    return (W, b)


# Sigmoid implementation
def sigmoid(z):
    return 1 / ( 1 + np.exp(z * -1))

# Forward propagation implementation
def forward_prop(weights, bias, input_data):
    Z = np.dot(weights.T, input_data) + bias
    return sigmoid(Z)

# Loss function implementation
def compute_loss(y_actual, y_predicted):
    epsilon = 1e-5
    inputSize = y_actual.shape[1]
    
    loss = -1/inputSize * np.sum(y_actual * np.log(y_predicted + epsilon) + (1 - y_actual) * np.log((1 - y_predicted) + epsilon))
    return loss

# Back propagation implementation
def back_prop(input_data, y_actual, y_predicted):
    inputSize = y_actual.shape[1]
    
    dz = y_predicted - y_actual
    db = 1/inputSize * np.sum(dz)
    dw = 1/inputSize * np.dot(input_data, dz.T)
    
    return (dz, db, dw)

# Update weights function implementation
def update_weights(weights, bias, dw, db, learning_rate): 
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    return (weights, bias)

# Function to train model
def train_model(epochs, learning_rate, input_data, input_labels):
    
    # Initialize weights
    W, b = initialize_parameters(input_data.shape)
    
    for i in range(epochs): 
        # Forward propagation
        predictions = forward_prop(W, b, input_data)
        
        # Compute loss
        loss = compute_loss(input_labels, predictions)
        
        # Debug: print loss every 300th epoch
        # if i % 300 == 0:
        #     print('Loss on epoch', i+1, 'is', loss)
        
        # Back propagation
        dz, db, dw = back_prop(input_data, input_labels, predictions)
        
        # Update weights
        W, b = update_weights(W, b, dw, db, learning_rate)


    # Debug
    # print('Final loss: ', loss)
        
    # Comment for game
    print('Done! If I were you, I would be about', 100 - ((loss * 100).round()), '% confident in my mushroom choices.')

    # Return final weights and bias
    return W, b

# Train model
print('Performing mushroom math... please be patient!')
final_weights, final_bias = train_model(5000, .58, mushroom_data_train, mushroom_labels_train)


# Function to predict on new input
def predict(new_input, weights, bias):
    predictions = forward_prop(weights, bias, new_input)
    return predictions

##########################################################
# GAME                                                   #
##########################################################

mushroomArt = '''
        __.....__
     .'" _  o    "`.
   .' O (_)     () o`.
  .           O       .
 . ()   o__...__    O  .
. _.--"""       """--._ .
:"                     ";       Mushroom Forest
 `-.__    :   :    __.-'
      """-:   :-"""
         J     L
         :     :
        J       L
        :       :
        `._____.' mh
'''

turnSeparator = '\n*******************************************************************************\n'
print(turnSeparator)
print(mushroomArt)
print(turnSeparator)

# Reference of possible values
possibleValues = {
    'cap-shape': [('bell', 'b'), ('conical', 'c'), ('convex', 'x'), ('flat', 'f'), ('knobbed', 'k'), ('sunken', 's')],
    'cap-surface': [('fibrous', 'f'), ('grooves', 'g'), ('scaly','y'), ('smooth','s')],
    'cap-color': [('brown', 'n'), ('buff', 'b'), ('cinnamon', 'c'), ('gray','g'), ('green','r'), ('pink','p'), ('purple','u'), ('red','e'), ('white','w'), ('yellow','y')],
    'bruises': [('It has bruises', 't'), ('It does not have bruises','f')], 
    'odor': [('almond','a'), ('anise','l'), ('creosote','c'), ('fishy','y'), ('foul','f'), ('musty','m'), ('none','n'), ('pungent','p'), ('spicy','s')], 
    'gill-attachment': [('attached','a'),('free','f')],
    'gill-spacing': [('close', 'c'), ('crowded', 'w')],
    'gill-size': [('broad', 'b'), ('narrow', 'n')],
    'gill-color': [('black','k'), ('brown','n'),('buff','b'),('chocolate','h'),('gray','g'),('green','r'),('orange','o'),('pink','p'),('purple','u'), ('red','e'), ('white','w'), ('yellow','y')],
    'stalk-shape': [('enlarging', 'e'), ('tapering', 't')],
    'stalk-root': [('bulbous','b'),('club','c'),('equal','e'),('rooted','r'),('unknown','?')],
    'stalk-surface-above-ring': [('fibrous', 'f'),('scaly', 'y'),('silky', 'k'),('smooth','s')],
    'stalk-surface-below-ring': [('fibrous', 'f'),('scaly', 'y'),('silky', 'k'),('smooth','s')],
    'stalk-color-above-ring': [('brown', 'n'), ('buff', 'b'), ('cinnamon', 'c'), ('gray','g'),('orange', 'o'),('pink', 'p'), ('red', 'e'), ('white', 'w'), ('yellow', 'y')],
    'stalk-color-below-ring': [('brown', 'n'), ('buff', 'b'), ('cinnamon', 'c'), ('gray','g'),('orange', 'o'),('pink', 'p'), ('red', 'e'), ('white', 'w'), ('yellow', 'y')],
    'veil-type': [('partial', 'p')],
    'veil-color': [('brown', 'n'),('orange', 'o'), ('white', 'w'), ('yellow', 'y')],
    'ring-number': [('none', 'n'), ('one', 'o'), ('two', 't')],
    'ring-type': [('evanescent','e'),('flaring','f'),('large','l'),('none','n'),('pendant','p')],
    'spore-print-color': [('black', 'k'),('brown', 'n'), ('buff', 'b'),('chocolate', 'h'),('green','r'),('orange','o'),('purple','u'),('white','w'), ('yellow','y')],
    'population': [('abundant','a'),('clustered','c'),('numerous','n'),('scattered','s'),('several','v'),('solitary','y')],
    'habitat': [('grasses','g'),('leaves','l'),('meadows','m'),('paths','p'),('urban','u'),('waste','w'),('woods','d')]
}

# Gameplay loop
isGameInProgress = True
hungerCount = random.randint(3, 5)

while isGameInProgress:
    # Generate random mushroom
    def genRandomMushroom(): 
        newMushroom = pd.DataFrame()
        print('You encounter a mushroom in your walk through the forest.') 
        print('You can\'t remember the last time you ate...')
        print('In fact, you\'re so hungry that you think you\'ll starve if you don\'t eat in the next', hungerCount, 'turns!')

        for attribute in possibleValues: 
            # Add info to dataframe
            randomValIdx = random.randint(0, len(possibleValues[attribute])-1)
            newMushroom[attribute] = [possibleValues[attribute][randomValIdx][1]]

            # For cap shape, print out the appropriate image
            if attribute == 'cap-shape':
                fileName = 'mushrooms/' + possibleValues[attribute][randomValIdx][0] + '.txt'
                mushroomImage = open(fileName).read()
                print(mushroomImage)

            # Print out for user to read
            if attribute == 'bruises': 
                print(possibleValues[attribute][randomValIdx][0])
            else: 
                nameList = attribute.split('-')
                nameStr = ' '.join(nameList).capitalize()
                print(nameStr, 'is', possibleValues[attribute][randomValIdx][0])

        # Return dataframe
        return newMushroom

    # Get dataframe corresponding to randomly generated mushroom, and convert its values to 
    # numbers using the learned encodings
    newMushroom = genRandomMushroom()

    for colName, colData in newMushroom.iteritems(): 
        convert_to_numbers(colData, numberMappings[colName])

    # Add cross features
    add_cross_features(newMushroom)

    # Get model prediction for new mushroom 
    newMushroom = newMushroom.to_numpy().T
    prediction = predict(newMushroom, final_weights, final_bias)
    if prediction >= .5:
        prediction = 1
    else:
        prediction = 0

    
    # Get user input for whether or not to eat the mushroom
    print("You've been walking long enough that you're quite hungry.")
    print("...But, mushrooms in these parts have a reputation for being poisonous.")
    didEat = ''

    while didEat.lower() != 'y' and didEat.lower() != 'n':
        didEat = input("Do you eat the mushroom? (Y/N)")

    # Evaluate how to update hunger/game state based on response and prediction
    if prediction == 0 and didEat.lower() == 'y':
        print("You aren't feeling so well...")
        isGameInProgress = False
        break
    elif prediction == 0 and didEat.lower() == 'n':
        print('It was poisonous! That was a close one.')

        hungerChance = random.randint(0,20)

        if hungerChance < 10:
            print('You\'re so proud of yourself for making the right choice that you haven\'t gotten any hungrier.')
        else:
            print('You\'re safe, but you still feel a little bit hungrier.')
            hungerCount -= 1

    elif prediction == 1 and didEat.lower() == 'y':
        print('You enjoyed the delicious mushroom!')
        hungerIncrease = random.randint(1, 3)
        hungerCount += hungerIncrease
        print('Your fullness level has increased by ' + str(hungerIncrease) + '!')
    elif prediction == 1 and didEat.lower() == 'n':
        print('You skipped out on a safe mushroom.')
        hungerDecrease = random.randint(1, 3)
        hungerCount -= hungerDecrease

        if hungerCount <= 0:
            print('You\'re so hungry you can\'t move...')
            isGameInProgress = False
            break
        
        print('Your fullness level has decreased by ' + str(hungerDecrease) + '...')
    
    print(turnSeparator)


# On game over, print a mushroom fun fact!
funFacts = ['Cultures around the world have eaten or used mushrooms medicinally for centuries, dating all the way back to ancient Egypt.', 
'More than 75 species of bioluminescent mushrooms exist on Earth.',
'Mushrooms are approximately: 50% inedible but harmless, 25% edible, but not incredible, 20% will make you sick, 4% will be tasty to excellent, 1% can kill you.',
'No one knows how many types of mushrooms exist in nature.',
'Mushrooms are made up of around 90% water.',
'In the Blue Mountains of Oregon is a colony of Armillaria solidipes that is believed to be the world’s largest known organism. The fungus is over 2,400 years old and covers an estimated 2,200 acres!',
'Before the invention of synthetic dyes, mushrooms were widely used for dyeing wool and other natural fibers.'
]

print(turnSeparator)
print('Bad news: you lost.')
print('Good news: here is a mushroom fun fact!')
print(funFacts[random.randint(0, len(funFacts)-1)])