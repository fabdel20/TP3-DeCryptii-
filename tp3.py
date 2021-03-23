# Farida Abdelmoneum's 15-112 TP -- deCrypttii
# Section D0 - fabdelmo
# TP mentor Lauren Sands

from cmu_112_graphics import * 
# Ceasar Cipher, Monoalphabetic Cipher, Homophonic Substitution Cipher,
# Polygram Substitution Cipher, Play Fair Cipher (encoding and decoding)
import random 
import string 
import math
import random

# Ceasar Cipher : 
# this simply shifts over the letters by however many is specified.
# Depending on the level it can be harder or easier(the shift) but this is for
# sure the easiest cipher 
# this code is from my hw3a 
def applyCaesarCipher(message, shift):
    result = "" 
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    newMessage = ""
    for i in range(len(message)):
        char = message[i].lower()
        if char in alphabet:
            newMessage += char
    for i in newMessage:
        val = ord(i)
        newVal = val + shift
        alphabetRange = 26
        #if i is a white space
        if (val == ord(" ")):
            result += chr(val)
        #if letter in message is a lowercase letter then 
        if (val >= ord('a') and val <= ord('z')):
            if(newVal > ord('z')):
                wrapAround = newVal - alphabetRange
                result += chr(wrapAround)
            #if the new ord value is greater than the bounds it wraps around
            elif(newVal < ord('a') and newVal > ord('Z')):
                wrapAround = newVal + alphabetRange
                result += chr(wrapAround)
            else:
                result += chr(newVal)
        #if the letter is uppercase
        if(val >= ord('A') and val <= ord('Z')):
            if(newVal > ord('Z')):
                wrapAround2 = newVal - alphabetRange
                result += chr(wrapAround2)
            elif(newVal < ord('A')):
                wrapAround3 = newVal + alphabetRange
                result += chr(wrapAround3)
            else:
                result += chr(newVal)
        
    return result

def testApplyCaesarCipher():
    print('Testing applyCaesarCipher()...', end='')
    assert(applyCaesarCipher('abcdefghijklmnopqrstuvwxyz', 3) ==
                             'defghijklmnopqrstuvwxyzabc')
    assert(applyCaesarCipher('We Attack At Dawn', 1) == 'Xf Buubdl Bu Ebxo')
    assert(applyCaesarCipher('1234', 6) == '1234')
    assert(applyCaesarCipher('abcdefghijklmnopqrstuvwxyz', 25) ==
                             'zabcdefghijklmnopqrstuvwxy')
    assert(applyCaesarCipher('We Attack At Dawn', 2)  == 'Yg Cvvcem Cv Fcyp')
    assert(applyCaesarCipher('We Attack At Dawn', 4)  == 'Ai Exxego Ex Hear')
    assert(applyCaesarCipher('We Attack At Dawn', -1) == 'Vd Zsszbj Zs Czvm')
    print('Passed.')
# print(testApplyCaesarCipher())

# Monoalphabetic Cipher -- this is a specific case cipher (ROT13)
# for this cipher, each letter is encrypted with another letter so for
# example every A that would be in the orignal text would be encrypted to a D
# this version of the monoalphabetic cipher is based on ROT 13 which is a common 
# cipher in the web, the key is shifted 13 spots but it is a subsitution cipher.
# to make sure my test cases were correct I used a ROT13 enocder/decoder website.
''' https://cryptii.com/pipes/caesar-cipher ''' 
# this is encoding ROT13
def ROT13(word):
    key = 'nopqrstuvwxyzabcdefghijklm'
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = ""

    for char in range(len(word)):
        letter = word[char]
        nL = letter.lower()
        if nL in key:
            # if the original message was in lower case it goee through this if-statement
            # to check the corresponding index in the key then get it from the alphabet
            if letter.isupper() == False:
                keyIndex = key.find(letter)
                newLetter = alphabet[keyIndex]
                result += newLetter
            # if the original message letter is in upper case it goes through this else-statement 
            else:
                lower = letter.lower()
                kIndex = key.find(lower)
                lowerL = alphabet[kIndex]
                upperL = lowerL.upper()
                result += upperL
        # this else statement is to help preserve anything that is not a letter
        else:
            result += letter
    return result 

def testROT13():
    print('Testing ROT13()...', end='')
    assert(ROT13('abcdefghijklmnopqrstuvwxyz') == 'nopqrstuvwxyzabcdefghijklm')
    assert(ROT13('We Attack At Dawn') == 'Jr Nggnpx Ng Qnja')
    assert(ROT13('1234') == '1234')
    print('Passed.')
# print(testROT13())

# Decoding ROT13 
def decodeROT13(word):
    key = 'nopqrstuvwxyzabcdefghijklm'
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = ""

    for char in range(len(word)):
            letter = word[char]
            nL = letter.lower()
            if nL in key:
                # if the letter is lower case, we will reverse what we did when 
                # encoding the cipher 
                if letter.isupper() == False:
                    keyIndex = alphabet.find(letter)
                    newLetter = key[keyIndex]
                    result += newLetter
                # if the letter is upper case, we will reverse what we did when 
                # encoding the cipher 
                else:
                    lower = letter.lower()
                    kIndex = alphabet.find(lower)
                    lowerL = key[kIndex]
                    upperL = lowerL.upper()
                    result += upperL
            # this else statement is to help preserve anything that is not a letter
            else:
                result += letter
    return result 

def testDecodeROT13():
    print('Testing decodeROT13()...', end='')
    assert(decodeROT13('nopqrstuvwxyzabcdefghijklm') == 'abcdefghijklmnopqrstuvwxyz')
    assert(decodeROT13('Jr Nggnpx Ng Qnja') == 'We Attack At Dawn')
    assert(decodeROT13('1234') == '1234')
    print('Passed.')
#print(testDecodeROT13())

# Monoalphabetic Cipher (General): 
# for this cipher, each letter is encrypted with another letter so for
# example every A that would be in the orignal text would be encrypted to a D 
def generateMonoalphabeticCipherKey():
    # this will generate a unique key everytime we do a monoalphabetic cipher
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    keyDictionary = dict()
    directoryOfKeysUsed = []
    for i in range(len(alphabet)):
        aLetter = alphabet[i]
        # picks a random number that corresponds with an ASCII value between A-Z
        ordVal = random.randrange(97,123)
        # if its not in the directory we added, else we generate a new number
        # this guarantees uniqueness 
        if ordVal not in directoryOfKeysUsed:
            letter = chr(ordVal)
            keyDictionary[aLetter] = letter
            directoryOfKeysUsed.append(ordVal)
        else:
            while ordVal in directoryOfKeysUsed:
                NordVal = random.randrange(97,123)
                ordVal = NordVal
            if ordVal not in directoryOfKeysUsed:
                letter = chr(ordVal)
                keyDictionary[aLetter] = letter
                directoryOfKeysUsed.append(ordVal)             
    return keyDictionary

def monoAlphabeticCipher(word):
    key = generateMonoalphabeticCipherKey()
    result = "" 

    # follows the same pattern as ROT13 but with the unique key that was generated 
    for char in range(len(word)):
        letter = word[char]
        nL = letter.lower()
        if nL in key:
            if letter.isupper() == False:
                keyIndex = key[letter]
                result += keyIndex
            else:
                lower = letter.lower()
                kIndex = key[lower]
                result += kIndex
        else:
            result += letter
    return result , key
#print(generateMonoalphabeticCipherKey())
#print(monoAlphabeticCipher("Hello my friends, what's up"))

# Monoalphabetic Cipher Decoder
def monoAlphabeticCipherDecoder(word, Key):
    # list comprehension of inverted key comes from 
    # https://dev.to/renegadecoder94/how-to-invert-a-dictionary-in-python-2150
    if len(Key) == 0:
        pass 
    prep = dict()
    count = 0
    for i in range(len(Key) // 2):
        if i < len(Key) - 1:
            if i == 0:
                l1 , l2 = Key[i], Key[i + 1]
                prep[l1] = l2 
                count += 1
            if i>0:
                l1, l2 = Key[i + count], Key[i + (count +1)]
                prep[l1] = l2
                count += 1
    invertedKey = {value : k for k, value in prep.items()}
    result = "" 

    # finds the key associated with the value from the inverted dictionary 
    for char in range(len(word)):
        letter = word[char]
        nL = letter.lower()
        if nL in prep:
            if letter.isupper() == False:
                keyIndex = invertedKey.get(letter, 0 )
                result += keyIndex
            else:
                lower = letter.lower()
                kIndex = invertedKey.get(lower, 0 )
                result += kIndex
        else:
            result += letter
    return result 
#print(monoAlphabeticCipherDecoder("vf gs ulgd fj zlyfkl", {'a': 'l', 'b': 'x', 'c': 't', 'd': 'k', 'e': 'd', 'f': 'z', 'g': 'c', 'h': 'v', 'i': 'f', 'j': 'o', 'k': 'h', 'l': 'e', 'm': 'g', 'n': 'u', 'o': 'q', 'p': 'r', 'q': 'm', 'r': 'y', 's': 'j', 't': 'a', 'u': 'n', 'v': 'i', 'w': 'p', 'x': 'w', 'y': 's', 'z': 'b'}))

# Homophonic Cipher:
# this is the same as the monoalphabetic cipher but uses multiple mappings to 
# the same key

# generates a unique key with multiple mappings
def generateHomophonicCipherKey():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    keyDictionary = dict()
    directoryOfKeysUsed = []
    for i in range(len(alphabet)):
        count = 0
        setOFVals = []
        aLetter = alphabet[i]
        while count <= 2:
            ordVal = random.randrange(58,126)
            if ordVal not in directoryOfKeysUsed:
                letter = chr(ordVal)
                setOFVals.append(letter)
                keyDictionary[aLetter] = setOFVals
                directoryOfKeysUsed.append(ordVal)
            else:
                while ordVal in directoryOfKeysUsed:
                    NordVal = random.randrange(58,126)
                    if NordVal not in directoryOfKeysUsed:
                        letter = chr(ordVal)
                        setOFVals.append(letter)
                        keyDictionary[aLetter] = setOFVals
                        directoryOfKeysUsed.append(ordVal)
                    ordVal = NordVal
            count += 1 
    return keyDictionary

def homophonicCipher(word):
    key = generateHomophonicCipherKey()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = ""

    for char in range(len(word)):
        letter = word[char]
        if letter.lower() in alphabet:
            letterCount = word.count(letter)
            randNum = random.randrange(0, 2)
            lOK = key.get(letter, 0)
            if type(lOK) != int: 
                firstVal = str(lOK[randNum])
                result += firstVal
        else:
            result += letter 
    return result, key
#print(homophonicCipher("the roses are pretty"))

# Polygram Cipher Substitution - I have done the vigenere cipher which has 
# many combinations but the structure of the cipher is somewhat predictable 
'''https://www.youtube.com/watch?v=zNO4PTlg62k''' 
# this produces a string with the codeWord corresponding to every letter in the 
# word
def codeWordShift(word, codeWord):
    word = word.replace(" ", "")
    result = ""
    for i in range(len(word)):
        letter = word[i]
        length = len(codeWord)
        if i >= length:
            newI = i % length 
            replaceLetter = codeWord[newI]
            result += replaceLetter
        elif i <= length-1:
            rLetter = codeWord[i]
            result += rLetter
    return result 

# uses the codeWordShift to find the shift which is the ord of index of the 
# codeWord letter and gets added to the letter of the original word 
def polygramCipher(word, codeWord):
    shiftedMessage = (codeWordShift(word, codeWord)).lower()
    cWord = word
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    word = (word.replace(" ", "")).lower()
    strippedWord = ""
    for i in range(len(word)):
        c = word[i]
        if c in alphabet:
            strippedWord += c
    result = ""
    spacedResult = ''
    for i in range(len(strippedWord)):
        letter = strippedWord[i]
        if letter in alphabet:
            numWord = alphabet.index(letter)
            numShift = alphabet.index(shiftedMessage[i])
            newIndex = numShift + numWord
            while newIndex >= 26:
                newIndex -= 26
            result += alphabet[newIndex]

    listCWord = list(cWord)
    listResult = list(result)
    for i in range(len(listCWord)):
        var = listCWord[i]
        if var.lower() not in alphabet:
            spacedResult += var
            listResult.insert(i, var)
        else:
            if var.isupper():
                spacedResult += listResult[i].upper()
            else:
                spacedResult += listResult[i]
    return spacedResult

#print(polygramCipher("How is your evening? I have been so tired and have been unable to do anything except for 15-112.", 'code'))

# decryption of PolygramCipher 
def decodePolygramCipher(message, codeWord):
    codeWordString = codeWordShift(message, codeWord)
    cWord = message
    message = (message.replace(" ", "")).lower()
    result = ""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    spacedResult = ""
    strippedWord = ""
    for i in range(len(message)):
        c = message[i]
        if c in alphabet:
            strippedWord += c
    for i in range(len(strippedWord)):
        letter = strippedWord[i]
        if letter in alphabet:
            numWord = alphabet.index(letter)
            numShift = alphabet.index(codeWordString[i])
            newIndex = numWord - numShift 
            if newIndex < 0: 
                newIndex += 26
            result += alphabet[newIndex]

    listCWord = list(cWord)
    listResult = list(result)
    for i in range(len(listCWord)):
        var = listCWord[i]
        if var.lower() not in alphabet:
            spacedResult += var
            listResult.insert(i, var)
        else:
            if var.isupper():
                spacedResult += listResult[i].upper()
            else:
                spacedResult += listResult[i]
    return spacedResult
#print(decodePolygramCipher("Jcz mu mryt syipwqk? K vdzg phip gr xkfhh cbg lcjh fgsq ypoepg hr hq oqcvvlri saggdw jqf 15-112. ", 'code'))


# Play Fair Cipher:
# there are multiple compnents to this cipher. The first is to generate a board
# this board will ommit one letter and based upon this board a the encryption 
# will be made. The second component is stripping the word of white spaces and 
# making diving the letters into pairs, if there is an odd number then we add 
# an 'x' to the last pair to make it even. Then we need to draw out the boxes of
# each pair and using the rules get the encrypted letters which we input in the 
# 'result' which is the encrypted message the user will receive. 

def boardGenerator():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    newAlphabet = alphabet.replace('q', "")
    rows = 5
    cols = 5
    i = 0 
    # this is from the list comprehension in the 15112 notes 
    # https://www.cs.cmu.edu/~112/notes/notes-2d-lists.html#creating2dLists 
    # just my board = is from 15112 notes 
    board = [([0] * cols) for row in range(rows)]
    for row in range(rows):
        for col in range(cols):
            board[row][col] = newAlphabet[i]
            i += 1
    return board  
#print(boardGenerator())

def prepWord(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    strippedWord = word.replace(" ", "")
    result = ""
    count = 0
    if len(strippedWord) % 2 == 0:
        for i in range(len(strippedWord)//2):
            letter = strippedWord[1]
            pair = strippedWord[count : count + 2]
            if pair[0] == pair[1]:
                pair = pair[0] + 'x'
                result += pair + ' '
            else:
                result += pair + ' '
            count += 2
    else:
        strippedWord += 'x'
        for i in range(len(strippedWord)// 2):
            letter = strippedWord[i]
            pair = strippedWord[count : count+2]
            if pair[0] == pair[1]:
                pair = pair[0] + 'x'
                result += pair + ' '
            else:
                result += pair + ' '
            count += 2
    for i in range(len(result)):
        letter = result[i]
        if letter not in alphabet:
            result.replace(letter, "")
    return result

#print(prepWord('hello world'))

def transformer(firstCord, secondCord, board):
    height = abs(firstCord[0] - secondCord[0]) + 1
    width = abs(firstCord[1] - secondCord[1]) + 1 
    newBoard = [([0] * width) for row in range(height)]
    result = ""
    for row in range(len(newBoard)):
        for col in range(len(newBoard[0])):
            newBoard[row][col] = (row, col)
    if width > 1 and height > 1:
        if firstCord[1] < secondCord[1]:
            row, shift = newBoard[height - 1][width -1]
            fRow, fCol = firstCord[0], firstCord[1] + shift 
            sRow, sCol = secondCord[0],secondCord[1] - shift 
            result += board[fRow][fCol] + board[sRow][sCol]
        if firstCord[1] > secondCord[1]:
            row, shift = newBoard[height - 1][width -1]
            fRow, fCol = firstCord[0], firstCord[1] - shift 
            sRow, sCol = secondCord[0],secondCord[1] + shift 
            result += board[fRow][fCol] + board[sRow][sCol]
    if width == 1 and height > 1:
        row1, col1 = (firstCord[0] + 1, firstCord[1])
        row2, col2 = (secondCord[0] + 1, secondCord[1])
        if row1 > len(board) - 1:
            row1 = 0 
        if row2 > len(board) - 1:
            row2 = 0
        result += board[row1][col1] + board[row2][col2]
    if height == 1 and width > 1:
        row1, col1 = firstCord[0], firstCord[1] + 1
        row2, col2 = secondCord[0], secondCord[1] + 1
        if col1 > len(board[0]) - 1:
            col1 = 0
        if col2 > len(board[0]) - 1:
            col2 = 0 
        result += board[row1][col1] + board[row2][col2]
    return result

def PlayFairCipher(word):
    board = boardGenerator()
    word = prepWord(word).replace(' ',"")
    firstCord = (0, 0)
    secondCord = (0, 0)
    count = 0 
    result = ''
    for i in range(len(word)):
        letter = word[i]
        for row in range(len(board)):
            for col in range(len(board[0])):
                val = board[row][col]
                if val == letter and count == 0:
                    firstCord = (row, col)
                    count = 1
                elif val == letter and count == 1:
                    secondCord = (row, col)
                    result += transformer(firstCord, secondCord, board)
                    firstCord = (0,0)
                    secondCord = (0,0)
                    count = 0
    return result 
#print(PlayFairCipher("what's up my dudes"))

def playFairCipherDecorder(word):
    board = boardGenerator()
    firstCord = (0, 0)
    secondCord = (0, 0)
    count = 0 
    result = ''
    spacedResult = ""
    for i in range(len(word)):
        letter = word[i]
        for row in range(len(board)):
            for col in range(len(board[0])):
                val = board[row][col]
                if val == letter and count == 0:
                    firstCord = (row, col)
                    count = 1
                elif val == letter and count == 1:
                    secondCord = (row, col)
                    result += decoderTransformer(firstCord, secondCord, board)
                    firstCord = (0,0)
                    secondCord = (0,0)
                    count = 0
    for i in range(len(result) // 2):
        if i < len(result):
            first, second = result[count : count + 2]
            if second == 'x':
                spacedResult += first + first 
                count += 2
            else: 
                spacedResult += first + second
                count += 2
        if i == len(result):
            letter = result[i]
            if letter == 'x':
                spacedResult += ""
            else:
                spacedResult += letter
    return spacedResult  

def decoderTransformer(firstCord, secondCord, board):
    height = abs(firstCord[0] - secondCord[0]) + 1
    width = abs(firstCord[1] - secondCord[1]) + 1 
    newBoard = [([0] * width) for row in range(height)]
    result = ""
    for row in range(len(newBoard)):
        for col in range(len(newBoard[0])):
            newBoard[row][col] = (row, col)
    if width > 1 and height > 1:
        if firstCord[1] > secondCord[1]:
            row, shift = newBoard[height - 1][width -1]
            fRow, fCol = firstCord[0], firstCord[1] - shift 
            sRow, sCol = secondCord[0],secondCord[1] + shift 
            result += board[fRow][fCol] + board[sRow][sCol]
        if firstCord[1] < secondCord[1]:
            row, shift = newBoard[height - 1][width -1]
            fRow, fCol = firstCord[0], firstCord[1] + shift 
            sRow, sCol = secondCord[0],secondCord[1] - shift 
            result += board[fRow][fCol] + board[sRow][sCol]
    if width == 1 and height > 1:
        row1, col1 = (firstCord[0] - 1, firstCord[1])
        row2, col2 = (secondCord[0] - 1, secondCord[1])
        if row1 > len(board) - 1:
            row1 = len(board) - 1 
        if row2 > len(board) - 1:
            row2 = len(board) - 1
        result += board[row1][col1] + board[row2][col2]
    if height == 1 and width > 1:
        row1, col1 = firstCord[0], firstCord[1] - 1
        row2, col2 = secondCord[0], secondCord[1] - 1
        if col1 > len(board[0]) - 1:
            col1 = len(board[0]) 
        if col2 > len(board[0]) - 1:
            col2 = len(board[0]) 
        result += board[row1][col1] + board[row2][col2]
    return result

#print(playFairCipherDecorder('xgdptpskditecu'))

# Hill Cipher
# the hill cipher is a polgram subsitituition cipher that uses linear algebra
# (this is what makes it algorithmically complex). How it works is that based
# on the length of the encrypted message(that is stripped of white spaces and 
# any extra characters) a matrix is formed with random numbers modulo 26. This
# matrix represents the key for this cipher. Then a vector is created with 
# the message, each letter is then input as modul 26(its position in the
# alphabet) Then we multiply out the vector and matrix and then do that modulo 
# 26. We then get a vector which we translate back into letters to get the 
# encrypted message. 
# to understand how this cipher works I refrenced the website bellow 
'''https://www.geeksforgeeks.org/hill-cipher/'''
def matrixMultiplication(keyMatrix, messageVector):
    resultVector = []
    for row in range(len(keyMatrix)):
        index = 0
        summation = 0 
        for col in range(len(keyMatrix[0])):
            val = keyMatrix[row][col] * messageVector[index][0]
            index += 1
            summation += val
        resultVector += [summation]
    return resultVector

def hillCipher(message):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    message = message.replace(' ', "")
    strippedMessage = ""
    for i in range(len(message)):
        lower = message[i].lower()
        if lower in alphabet:
            strippedMessage += lower
    rows = cols = len(message)
    keyMatrix = [([0] * cols) for row in range(rows)]
    for row in range(len(keyMatrix)):
        for col in range(len(keyMatrix[0])):
            keyMatrix[row][col] = random.randrange(0,26)
    messageVector = [([0] * 1) for row in range(len(message))]
    for row in range(len(messageVector)):
        for col in range(len(messageVector[0])):
            letter = message[row]
            index = alphabet.find(letter)
            messageVector[row][col] = index
    newVector = matrixMultiplication(keyMatrix, messageVector)
    modVector = []
    for val in range(len(newVector)):
        num = newVector[val] % 26
        modVector += [num]
    result = ""
    for num in range(len(modVector)):
        index = modVector[num]
        result += alphabet[index]
    return result
#print(hillCipher("hello my name is farida:)"))
    
# This is the code to decrypt the Hill Cipher 
# https://stackoverflow.com/questions/4287721/easiest-way-to-perform-modular-matrix-inversion-with-python
# I used this code to find the modular inverse of a matrix but the matrix 
# multiplication function is my own and is modified from the first matrix
# multiplication function I made for enocding because of the nature of the
# moduler inverse matrix, the decryption algorithm is also my own. 
# learned how to import numpy and access it from the website below
# https://numpy.org/install/
import numpy
import math
from numpy import matrix
from numpy import linalg

def modMatInv(A,p):       # Finds the inverse of matrix A mod p
  n=len(A)
  A=matrix(A)
  adj=numpy.zeros(shape=(n,n))
  for i in range(0,n):
    for j in range(0,n):
      adj[i][j]=((-1)**(i+j)*int(round(linalg.det(minor(A,j,i)))))%p
  return (modInv(int(round(linalg.det(A))),p)*adj)%p

def modInv(a,p):          # Finds the inverse of a mod p, if it exists
  for i in range(1,p):
    if (i*a)%p==1:
      return i
  raise ValueError(str(a)+" has no inverse mod "+str(p))

def minor(A,i,j):    # Return matrix A with the ith row and jth column deleted
  A=numpy.array(A)
  minor=numpy.zeros(shape=(len(A)-1,len(A)-1))
  p=0
  for s in range(0,len(minor)):
    if p==i:
      p=p+1
    q=0
    for t in range(0,len(minor)):
      if q==j:
        q=q+1
      minor[s][t]=A[p][q]
      q=q+1
    p=p+1
  return minor

def matrixMultiplicationI(keyMatrix, messageVector):
    resultVector = []
    for row in range(len(keyMatrix)):
        index = 0
        summation = 0 
        for col in range(len(keyMatrix[0])):
          val = keyMatrix[row][col] * messageVector[index]
          index += 1
          summation += val
        resultVector.append(summation)
    return resultVector

def decodeHillCipher(eMessage, key):
    inverseMatrix = modMatInv(key, 26)
    rows = len(inverseMatrix)
    cols = len(inverseMatrix[0])
    # list comprehension from https://www.cs.cmu.edu/~112/notes/notes-2d-lists.html
    nm = [([0] * cols) for row in range(rows)]
    for r in range(len(inverseMatrix)):
      for c in range(len(inverseMatrix[0])):
        num = int(inverseMatrix[r][c])
        nm[r][c] = num
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    eMessageVector = []
    for char in range(len(eMessage)):
        letter = eMessage[char]
        index = alphabet.find(letter)
        eMessageVector += [index]
    newVector = matrixMultiplicationI(nm, eMessageVector)
    resultMatrix = []
    for row in range(len(newVector)):
      num = newVector[row]
      resultMatrix += [num%26]
    result = ""
    for row in range(len(resultMatrix)):
      index = resultMatrix[row]
      result += alphabet[index]
    return result 

sentences = ['hello world', 'welcome to decryptii',
            "coding is fun", "encryption", "decryption",
            "CMU is a good school", "scotty the tartan", "a tartan's responsibility",
            "covid sucks", "quarentine", "ice cream is good",
            "tennis is fun", "nutrition is important",
            "fastfood is bad for you", "the weather is good", "the sun is out",
            "candy tastes bad", "chocolate tastes good", "swimming is fun", "going to the beach is fun",
            "computer science", "machine learning", "electrical engineering",
            "dietriech college", "college of fine arts", 'school of computer science', 'college of engineering',
            "mellon college of sciences", "donner", "resnick", "res on fifth", "pittsburgh",
            "pennsylvania", "this game is cmu themed", "fun stuff"]

codeWords = ['code', 'fun', 'sun', 'cold', 'nice', 'summer', 'secret', 'cool']
        
class myIntroPage(Mode):
    def appStarted(self):
        self.message = '!gnirehpic ot emoclew'
        self.phrase = ''

    def mousePressed(self, event):
        self.phrase = self.getUserInput('Try to decode the message!')
        if (self.phrase == 'welcome to ciphering!'):
            self.message = 'Welcome!'
            self.app.setActiveMode(self.app.levelPage)
        while self.phrase != 'welcome to ciphering!':
            self.message = ('!gnirehpic ot emoclew')
            self.phrase = self.getUserInput('Try to decode the message!')
            if self.phrase == 'welcome to ciphering!':
                self.app.setActiveMode(self.app.levelPage)

    def redrawAll(self, canvas):
        font = 'Arial 24 bold'
        canvas.create_text(self.width/2,  self.height/2,
                           text=self.message, font=font)
        if (self.phrase != 'Welcome to Ciphering') and (self.phrase != ''):
            canvas.create_text(self.width/2, (self.height/2) - 40, text = "Try Again!", font = 'Arial 25 bold')
        

class instructionPage(Mode):
    def introPage(app, canvas):
        wCenter = app.width / 2
        hCenter = app.height / 2
        canvas.create_rectangle(0 , 0, app.width, app.height, fill = 'light blue')
        canvas.create_rectangle(wCenter - (0.5 * wCenter), hCenter - (0.5 * hCenter)
                                ,wCenter + (0.5 * wCenter),
                                hCenter + (0.5 * hCenter),
                                fill = 'blue')
        canvas.create_text(wCenter, 250, fill = 'white', text = "Welcome",
                                font = "Arial 26 bold")
        canvas.create_rectangle(wCenter - 190, hCenter - 110, wCenter - 60, hCenter-20, fill = 'red')
        canvas.create_text(wCenter + 70, hCenter - 65, text = 'Press the red button or press\nthe space bar to play deCryptii', font = 'Arial 16 bold', fill = 'white')
        canvas.create_rectangle(wCenter - 190, hCenter + 40, wCenter - 60, hCenter + 130, fill = 'light green')
        canvas.create_text(wCenter + 70, hCenter + 85, text = "Press the green button or 'p'\nto go to practice mode", font = 'Arial 18 bold', fill = 'white')
        canvas.create_text(wCenter, 50, text = "DeCryptii", fill = 'White',
                                font = "Arial 65 bold")
    def keyPressed(mode, event):
        if event.key == 'Space':
            mode.app.setActiveMode(mode.app.myIntroPage)
        if event.key == 's':
            mode.app.setActiveMode(mode.app.levelPage)
        if event.key == 'p':
            mode.app.setActiveMode(mode.app.practiceMode)
    
    def mousePressed(app, event):
        if ((app.width / 2) - 190) <= event.x <= ((app.width / 2) - 60) and ((app.height / 2) - 110) <= event.y <= ((app.height / 2)-20):
            app.app.setActiveMode(app.app.myIntroPage)
        if ((app.width / 2) - 190) <= event.x <= ((app.width / 2) - 60) and ((app.height / 2) +40) <= event.y <= ((app.height / 2)+ 130):
            app.app.setActiveMode(app.app.practiceMode)
    
    def redrawAll(self,canvas):
        self.introPage(canvas)

class levelPage(Mode):
    def appStarted(app):
        app.lives = app.app.lives
        app.level = app.app.level

    def title(app,canvas):
        canvas.create_rectangle(0 , 0, app.width, app.height, fill = 'light blue')
        canvas.create_rectangle(app.width/2 - app.width/8, 10, app.width/2 + app.width/8, 80, fill = 'red')
        canvas.create_text(app.width/2, 40, text = 'Levels', fill = 'Black', font = "Arial 35 bold")
    
    def keyPressed(app,event):
        if event.key =='u':
            app.app.level = 49
        if event.key == 'x':
            app.app.setActiveMode(app.app.instructionPage)

    def levelBoxes(app,canvas):
        canvas.create_rectangle(app.width/4 - app.width/3, 100, app.width/8 , 180, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/3, 200, app.width/8 , 280, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/3, 300, app.width/8 , 380, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/3, 400, app.width/8 , 480, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/3, 500, app.width/8 , 580, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/3, 600, app.width/8 , 680, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/3, 700, app.width/8 , 780, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 100, app.width/3.8 , 180, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 200, app.width/3.8 , 280, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 300, app.width/3.8 , 380, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 400, app.width/3.8 , 480, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 500, app.width/3.8 , 580, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 600, app.width/3.8 , 680, fill = 'red')
        canvas.create_rectangle(app.width/8 + 10, 700, app.width/3.8 , 780, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 100, app.width/2.5 , 180, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 200, app.width/2.5 , 280, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 300, app.width/2.5 , 380, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 400, app.width/2.5 , 480, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 500, app.width/2.5 , 580, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 600, app.width/2.5 , 680, fill = 'red')
        canvas.create_rectangle(app.width/3.8 + 10, 700, app.width/2.5 , 780, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 100, app.width/2 + 30 , 180, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 200, app.width/2 + 30 , 280, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 300, app.width/2 + 30 , 380, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 400, app.width/2 + 30 , 480, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 500, app.width/2 + 30 , 580, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 600, app.width/2 + 30 , 680, fill = 'red')
        canvas.create_rectangle(app.width/2.5 + 10, 700, app.width/2 + 30 , 780, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 100, app.width/2 + 145 , 180, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 200, app.width/2 + 145 , 280, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 300, app.width/2 + 145 , 380, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 400, app.width/2 + 145 , 480, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 500, app.width/2 + 145 , 580, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 600, app.width/2 + 145 , 680, fill = 'red')
        canvas.create_rectangle(app.width/2 + 40, 700, app.width/2 + 145 , 780, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 100, app.width/2 + 260 , 180, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 200, app.width/2 + 260 , 280, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 300, app.width/2 + 260 , 380, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 400, app.width/2 + 260 , 480, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 500, app.width/2 + 260 , 580, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 600, app.width/2 + 260 , 680, fill = 'red')
        canvas.create_rectangle(app.width/2 + 155, 700, app.width/2 + 260 , 780, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 100, app.width/2 + 385 , 180, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 200, app.width/2 + 385 , 280, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 300, app.width/2 + 385 , 380, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 400, app.width/2 + 385 , 480, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 500, app.width/2 + 385 , 580, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 600, app.width/2 + 385 , 680, fill = 'red')
        canvas.create_rectangle(app.width/2 + 270, 700, app.width/2 + 385 , 780, fill = 'red')

        canvas.create_text((app.width/4 - app.width/8) - 50, 130, text = 'Level 1', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 130, text = 'Level 2', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width / 3.8 + 60, 130, text = 'Level 3', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 130, text = 'Level 4', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 130, text = 'Level 5', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 130, text = 'Level 6', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 130, text = 'Level 7', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 150, text = 'Caesar', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 230, text = 'Level 8', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 230, text = 'Level 9', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width / 3.8 + 60, 230, text = 'Level 10', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 230, text = 'Level 11', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 230, text = 'Level 12', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 230, text = 'Level 13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 230, text = 'Level 14', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 250, text = 'ROT13', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 330, text = 'Level 15', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text(app.width/4 - 40, 330, text = 'Level 16', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text(app.width / 3.8 + 60, 330, text = 'Level 17', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text(app.width/2.5 + 60, 330, text = 'Level 18', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text(app.width/2 + 90, 330, text = 'Level 19', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text(app.width/2 + 207, 330, text = 'Level 20', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text(app.width/2 + 330, 330, text = 'Level 21', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 350, text = 'Monoalphabetic', fill = 'Black', font = "Arial 12 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 430, text = 'Level 22', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/4 - 40, 430, text = 'Level 23', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width / 3.8 + 60, 430, text = 'Level 24', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2.5 + 60, 430, text = 'Level 25', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2 + 90, 430, text = 'Level 26', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2 + 207, 430, text = 'Level 27', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2 + 330, 430, text = 'Level 28', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 450, text = 'Homophonic', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 530, text = 'Level 29', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/4 - 40, 530, text = 'Level 30', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width / 3.8 + 60, 530, text = 'Level 31', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2.5 + 60, 530, text = 'Level 32', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2 + 90, 530, text = 'Level 33', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2 + 207, 530, text = 'Level 34', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text(app.width/2 + 330, 530, text = 'Level 35', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 550, text = 'Polygram', fill = 'Black', font = "Arial 15 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 630, text = 'Level 36', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 630, text = 'Level 37', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width / 3.8 + 60, 630, text = 'Level 38', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 630, text = 'Level 39', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 630, text = 'Level 40', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 630, text = 'Level 41', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 630, text = 'Level 42', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 650, text = 'Playfair', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 730, text = 'Level 43', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text((app.width/4 - app.width/8) - 50, 750, text = 'Hill', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 730, text = 'Level 44', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/4 - 40, 750, text = 'Hill', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width / 3.8 + 60, 730, text = 'Level 45', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/3.8 + 60, 750, text = 'Hill', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 730, text = 'Level 46', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2.5 + 60, 750, text = 'Hill', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 730, text = 'Level 47', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 90, 750, text = 'Hill', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 730, text = 'Level 48', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 207, 750, text = 'Hill', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 730, text = 'Level 49', fill = 'Black', font = "Arial 18 bold")
        canvas.create_text(app.width/2 + 330, 750, text = 'RSA', fill = 'Black', font = "Arial 18 bold")

    def levelBox(app,canvas):
        canvas.create_rectangle((app.width/4 + 80) - app.width/3, 10, app.width/8 + 40, 80, fill = 'red')
        canvas.create_text(app.width/4 - app.width/6, 40, text = f'Level: {app.app.level}', fill = 'Black', font = "Arial 25 bold")
    def livesBox(app,canvas):
        canvas.create_rectangle(app.width/2 + 230, 10, app.width/2 + 410, 80, fill = 'red')
        canvas.create_text(app.width/2 + 300, 40, text = f'Lives: {app.app.lives}', fill = 'Black', font = "Arial 25 bold")
    def practiceBox(app,canvas):
        canvas.create_rectangle(app.width/8 + 60, 10, app.width/2 - 120, 80, fill = 'yellow')
        canvas.create_text(app.width/8 + 120, 45, text = 'Practice\n  Mode', font = 'Times 25 bold', fill = 'black')
    
    
    def mousePressed(app,event):
        if (app.width/8 + 60) <= event.x <= (app.width/2 - 120) and 10 <= event.y <= 80:
            app.app.setActiveMode(app.app.practiceMode)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (100 <= event.y <= 180) and app.app.level >= 1:
            app.app.setActiveMode(app.app.level1)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (100 <= event.y <= 180) and app.app.level >= 2:
            app.app.setActiveMode(app.app.level2)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (100 <= event.y <= 180) and app.app.level >= 3:
            app.app.setActiveMode(app.app.level3)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (100 <= event.y <= 180) and app.app.level >= 4:
            app.app.setActiveMode(app.app.level4)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (100<= event.y <= 180) and app.app.level >= 5:
            app.app.setActiveMode(app.app.level5)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (100<= event.y <= 180) and app.app.level >= 6:
            app.app.setActiveMode(app.app.level6)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (100<= event.y <= 180) and app.app.level >= 7:
            app.app.setActiveMode(app.app.level7)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (200 <= event.y <= 280) and app.app.level >= 8:
            app.app.setActiveMode(app.app.level8)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (200 <= event.y <= 280) and app.app.level >= 9:
            app.app.setActiveMode(app.app.level9)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (200 <= event.y <= 280) and app.app.level >= 10:
            app.app.setActiveMode(app.app.level10)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (200 <= event.y <= 280) and app.app.level >= 11:
            app.app.setActiveMode(app.app.level11)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (200<= event.y <= 280) and app.app.level >= 12:
            app.app.setActiveMode(app.app.level12)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (200<= event.y <= 280) and app.app.level >=  13:
            app.app.setActiveMode(app.app.level13)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (200<= event.y <= 280) and app.app.level >= 14:
            app.app.setActiveMode(app.app.level14)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (300 <= event.y <= 380) and app.app.level >= 15:
            app.app.setActiveMode(app.app.level15)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (300 <= event.y <= 380) and app.app.level >= 16:
            app.app.setActiveMode(app.app.level16)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (300 <= event.y <= 380) and app.app.level >= 17:
            app.app.setActiveMode(app.app.level17)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (300 <= event.y <= 380) and app.app.level >= 18:
            app.app.setActiveMode(app.app.level18)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (300<= event.y <= 380) and app.app.level >= 19:
            app.app.setActiveMode(app.app.level19)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (300<= event.y <= 380) and app.app.level >= 20:
            app.app.setActiveMode(app.app.level20)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (300<= event.y <= 380) and app.app.level >=  21:
            app.app.setActiveMode(app.app.level21)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (400 <= event.y <= 480) and app.app.level >=  22:
            app.app.setActiveMode(app.app.level22)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (400 <= event.y <= 480) and app.app.level >=  23:
            app.app.setActiveMode(app.app.level23)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (400 <= event.y <= 480) and app.app.level >= 24:
            app.app.setActiveMode(app.app.level24)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (400 <= event.y <= 480) and app.app.level >= 25:
            app.app.setActiveMode(app.app.level25)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (400<= event.y <= 480) and app.app.level >= 26:
            app.app.setActiveMode(app.app.level26)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (400<= event.y <= 480) and app.app.level >= 27:
            app.app.setActiveMode(app.app.level27)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (400<= event.y <= 480) and app.app.level >= 28:
            app.app.setActiveMode(app.app.level28)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (500 <= event.y <= 580) and app.app.level >=  29:
            app.app.setActiveMode(app.app.level29)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (500 <= event.y <= 580) and app.app.level >=  30:
            app.app.setActiveMode(app.app.level30)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (500 <= event.y <= 580) and app.app.level >= 31:
            app.app.setActiveMode(app.app.level31)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (500 <= event.y <= 580) and app.app.level >=  32:
            app.app.setActiveMode(app.app.level32)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (500<= event.y <= 580) and app.app.level >= 33:
            app.app.setActiveMode(app.app.level33)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (500<= event.y <= 580) and app.app.level >= 34:
            app.app.setActiveMode(app.app.level34)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (500<= event.y <= 580) and app.app.level >=  35:
            app.app.setActiveMode(app.app.level35)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (600 <= event.y <= 680) and app.app.level >= 36:
            app.app.setActiveMode(app.app.level36)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (600 <= event.y <= 680) and app.app.level >=  37:
            app.app.setActiveMode(app.app.level37)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (600 <= event.y <= 680) and app.app.level >=  38:
            app.app.setActiveMode(app.app.level38)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (600 <= event.y <= 680) and app.app.level >= 39:
            app.app.setActiveMode(app.app.level39)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (600<= event.y <= 680) and app.app.level >=  40:
            app.app.setActiveMode(app.app.level40)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (600<= event.y <= 680) and app.app.level >=  41:
            app.app.setActiveMode(app.app.level41)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (600<= event.y <= 680) and app.app.level >= 42:
            app.app.setActiveMode(app.app.level42)
        if (app.width/4 - app.width/3 <= event.x < app.width/8) and (700 <= event.y <= 780) and app.app.level >= 43:
            app.app.setActiveMode(app.app.level43)
        if (app.width/8 + 10 <= event.x <= app.width/3.8) and (700 <= event.y <= 780) and app.app.level >= 44:
            app.app.setActiveMode(app.app.level44)
        if(app.width/3.8 + 10 <= event.x <= app.width/2.5) and (700 <= event.y <= 780) and app.app.level >= 45:
            app.app.setActiveMode(app.app.level45)
        if (app.width/2.5 + 10 <= event.x <= app.width/2 + 30) and (700 <= event.y <= 780) and app.app.level >= 46:
            app.app.setActiveMode(app.app.level46)
        if(app.width/2 + 40 <= event.x <= app.width/2 + 145) and (700<= event.y <= 780) and app.app.level >= 47:
            app.app.setActiveMode(app.app.level47)
        if(app.width/2 + 155<= event.x <= app.width/2 + 260) and (700<= event.y <= 780) and app.app.level >= 48:
            app.app.setActiveMode(app.app.level48)
        if(app.width/2 + 270 <= event.x <= app.width/2 + 385) and (700<= event.y <= 780) and app.app.level >= 49:
            app.app.setActiveMode(app.app.level49)

    def redrawAll(mode, canvas):
        mode.title(canvas)
        mode.levelBoxes(canvas)
        mode.levelBox(canvas)
        mode.livesBox(canvas)
        mode.practiceBox(canvas)

class gameOver(Mode):
    def appStarted(app):
        app.app.level = 1
        app.app.lives = 3 
    
    def gameOverMessage(app,canvas):
        message = "Oh no!\nGame Over!\nPress 'r' to restart game\nPress 'p' to practice\nPress 'v' to go to Visualizer"
        canvas.create_text(app.height/2, app.width/2, text = message, font = 'Arial 55 bold', fill = 'red')
    
    def keyPressed(app,event):
        if event.key == 'r':
            app.app.setActiveMode(app.app.levelPage)
        if event.key == 'v':
            app.app.setActiveMode(app.app.visualizerMode)
        if event.key == 'p':
            app.app.setActiveMode(app.app.practiceMode)
    
    def redrawAll(app,canvas):
        app.gameOverMessage(canvas)

class level1(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 1: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level2)
                if app.app.level < 2:
                    app.app.level = 2
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level2)
                    if app.app.level < 2:
                        app.app.level = 2
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

            
class level2(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 2: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level3)
                if app.app.level < 3:
                    app.app.level = 3
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level3)
                    if app.app.level < 3:
                        app.app.level = 3
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level3(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 3: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level4)
                if app.app.level < 4:
                    app.app.level = 4
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level4)
                    if app.app.level < 4:
                        app.app.level = 4
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level4(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 4: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level5)
                if app.app.level < 5:
                    app.app.level = 5
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level5)
                    if app.app.level < 5:
                        app.app.level = 5
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level5(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 5: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level6)
                if app.app.level < 6:
                    app.app.level = 6
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level5)
                    if app.app.level < 6:
                        app.app.level = 6
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level6(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 6: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level7)
                if app.app.level < 7:
                    app.app.level = 7
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level7)
                    if app.app.level < 7:
                        app.app.level = 7
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level7(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = applyCaesarCipher(app.message, app.shift)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 7: Decrypt Caesar Cipher', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = {app.shift}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Caesar's cipher, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level8)
                if app.app.level < 8:
                    app.app.level = 8
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level8)
                    if app.app.level < 8:
                        app.app.level = 8
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level8(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = ROT13(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 8: Decrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = 13', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using ROT13, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level9)
                if app.app.level < 9:
                    app.app.level = 9
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level9)
                    if app.app.level < 9:
                        app.app.level = 9
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level9(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 9: Encrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using ROT13, first input a message\nthen input your encrypted message using ROT13.\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
        
    def keyPressed(app,event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptROT13 = decodeROT13(userMessage)
            if (encryptedMessage == decryptROT13):
                app.app.setActiveMode(app.app.level10)
                if app.app.level < 10:
                    app.app.level = 10
                    app.app.lives = 3
            if encryptedMessage == None or userMessage == None:
                pass 
            while encryptedMessage != decryptROT13 and userMessage != None and encrtyptedMessage != None and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptROT13:
                    app.app.setActiveMode(app.app.level10)
                    if app.app.level < 10:
                        app.app.level = 10
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level10(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = ROT13(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 10: Decrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = 13', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using ROT13, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')
        
    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level11)
                if app.app.level < 11:
                    app.app.level = 11
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level11)
                    if app.app.level < 11:
                        app.app.level = 11
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level11(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 11: Encrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using ROT13, first input a message\nthen input your encrypted message using ROT13.\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
        
    def keyPressed(app,event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptROT13 = decodeROT13(userMessage)
            if (encryptedMessage == decryptROT13):
                app.app.setActiveMode(app.app.level12)
                if app.app.level < 12:
                    app.app.level = 12
                    app.app.lives = 3
            if encryptedMessage == None or userMessage == None:
                pass 
            while encryptedMessage != decryptROT13 and userMessage != None and encrtyptedMessage != None and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptROT13:
                    app.app.setActiveMode(app.app.level12)
                    if app.app.level < 12:
                        app.app.level = 12
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level12(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = ROT13(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 12: Decrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = 13', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using ROT13, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')
        
    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level13)
                if app.app.level < 13:
                    app.app.level = 13
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level3)
                    if app.app.level < 13:
                        app.app.level = 13
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level13(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 13: Encrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using ROT13, first input a message\nthen input your encrypted message using ROT13.\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
        
    def keyPressed(app,event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptROT13 = decodeROT13(userMessage)
            if (encryptedMessage == decryptROT13):
                app.app.setActiveMode(app.app.level14)
                if app.app.level < 14:
                    app.app.level = 14
                    app.app.lives = 3
            if encryptedMessage == None or userMessage == None:
                pass 
            while encryptedMessage != decryptROT13 and userMessage != None and encrtyptedMessage != None and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptROT13:
                    app.app.setActiveMode(app.app.level14)
                    if app.app.level < 14:
                        app.app.level = 14
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
            
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level14(Mode):
    def appStarted(app):
        app.shift = random.randrange(1,25)
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = ROT13(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 14: Decrypt ROT13', font = 'Arial 50 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def alphabet(app, canvas):
        app.alphabetLetters = 'A B C D E F G H I J K L M N O P Q R S T U W X Y Z'
        canvas.create_text(app.width/2, app.height/2 + 50, text = f'{app.alphabetLetters}', font = 'Arial 25 bold')
    
    def shiftText(app, canvas):
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'Shift = 13', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using ROT13, decrypt the \nbolded message below.The alphabet\nalong with the shift number is provided to\nhelp you get started. Press 's' to start"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')
        
    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level15)
                if app.app.level < 15:
                    app.app.level = 15
                    app.app.lives = 3
            if phrase == None:
                pass 
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level15)
                    if app.app.level < 15:
                        app.app.level = 15
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.alphabet(canvas)
        app.shiftText(canvas)
        app.instructions(canvas)

class level15(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = monoAlphabeticCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 15: Decrypt Monoalphabetic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        sec1 = key[0:110]
        sec2 = key[110:200]
        sec3 = key[200:]
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'{sec1}\n{sec2}\n{sec3}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.\nPress 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level16)
                if app.app.level < 16:
                    app.app.level = 16
                    app.app.lives = 3
            if (phrase == None):
                pass
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level16)
                    if app.app.level < 16:
                        app.app.level = 16
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)


class level16(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 16: Encrypt Monoalphabetic Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, encrypt\na message and provide a key\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start!\nInput x into every text box to exit level"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def keyPressed(app, event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            key = app.getUserInput('Enter your key, Example: acbd, c is assigned to a and d is assigned to b, do this for the entire alphabet')
            if key == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if userMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if encryptedMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if (encryptedMessage == (monoAlphabeticCipherDecoder(encryptedMessage, key))) and userMessage != 'x' and key != 'x' and encryptedMessage != 'x':
                app.app.setActiveMode(app.app.level17)
                if app.app.level < 17:
                    app.app.level = 17
                    app.app.lives = 3
            while encryptedMessage != (monoAlphabeticCipherDecoder(encryptedMessage, key)) and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == (monoAlphabeticCipherDecoder(encryptedMessage, key)):
                    app.app.setActiveMode(app.app.level17)
                    if app.app.level < 17:
                        app.app.level = 17
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)

    def mousePressed(app,event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level17(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = monoAlphabeticCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 17: Decrypt Monoalphabetic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        sec1 = key[0:110]
        sec2 = key[110:200]
        sec3 = key[200:]
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'{sec1}\n{sec2}\n{sec3}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.\nPress 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level18)
                if app.app.level < 18:
                    app.app.level = 18
                    app.app.lives = 3
            if (phrase == None):
                pass
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level18)
                    if app.app.level < 18:
                        app.app.level = 18
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level18(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 18: Encrypt Monoalphabetic Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, encrypt\na message and provide a key\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start!\nInput x into every text box to exit level"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def keyPressed(app, event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            key = app.getUserInput('Enter your key, Example: acbd, c is assigned to a and d is assigned to b, do this for the entire alphabet')
            if key == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if userMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if encryptedMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if (encryptedMessage == (monoAlphabeticCipherDecoder(encryptedMessage, key))) and userMessage != 'x' and key != 'x' and encryptedMessage != 'x':
                app.app.setActiveMode(app.app.level19)
                if app.app.level < 19:
                    app.app.level = 19
                    app.app.lives = 3
            while encryptedMessage != (monoAlphabeticCipherDecoder(encryptedMessage, key)) and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == (monoAlphabeticCipherDecoder(encryptedMessage, key)):
                    app.app.setActiveMode(app.app.level19)
                    if app.app.level < 19:
                        app.app.level = 19
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)

    def mousePressed(app,event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level19(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = monoAlphabeticCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 19: Decrypt Monoalphabetic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        sec1 = key[0:110]
        sec2 = key[110:200]
        sec3 = key[200:]
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'{sec1}\n{sec2}\n{sec3}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.\nPress 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level20)
                if app.app.level < 20:
                    app.app.level = 20
                    app.app.lives = 3
            if (phrase == None):
                pass
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level20)
                    if app.app.level < 20:
                        app.app.level = 20
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level20(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 20: Encrypt Monoalphabetic Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, encrypt\na message and provide a key\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start!\nInput x into every text box to exit level"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def keyPressed(app, event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            key = app.getUserInput('Enter your key, Example: acbd, c is assigned to a and d is assigned to b, do this for the entire alphabet')
            if key == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if userMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if encryptedMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            if (encryptedMessage == (monoAlphabeticCipherDecoder(encryptedMessage, key))) and userMessage != 'x' and key != 'x' and encryptedMessage != 'x':
                app.app.setActiveMode(app.app.level21)
                if app.app.level < 21:
                    app.app.level = 21
                    app.app.lives = 3
            while encryptedMessage != (monoAlphabeticCipherDecoder(encryptedMessage, key)) and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == (monoAlphabeticCipherDecoder(encryptedMessage, key)):
                    app.app.setActiveMode(app.app.level21)
                    if app.app.level < 21:
                        app.app.level = 21
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)

    def mousePressed(app,event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level21(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = monoAlphabeticCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 21: Decrypt Monoalphabetic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        sec1 = key[0:110]
        sec2 = key[110:200]
        sec3 = key[200:]
        canvas.create_text(app.width/2, app.height/2 + 80, text = f'{sec1}\n{sec2}\n{sec3}', font = 'Arial 25')
    
    def instructions(app, canvas):
        text = "Using Monoalphabetic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.\nPress 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)
    
    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level22)
                if app.app.level < 22:
                    app.app.level = 22
                    app.app.lives = 3
            if (phrase == None):
                pass
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level22)
                    if app.app.level < 22:
                        app.app.level = 22
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level22(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 22: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level23)
                if app.app.level < 23:
                    app.app.level = 23
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level23)
                    if app.app.level < 23:
                        app.app.level = 23
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level23(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 23: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level24)
                if app.app.level < 24:
                    app.app.level = 24
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level24)
                    if app.app.level < 24:
                        app.app.level = 24
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)


class level24(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 24: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level25)
                if app.app.level < 25:
                    app.app.level = 25
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level25)
                    if app.app.level < 25:
                        app.app.level = 25
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level25(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 25: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level26)
                if app.app.level < 26:
                    app.app.level = 26
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level26)
                    if app.app.level < 26:
                        app.app.level = 26
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)


class level26(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 26: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level27)
                if app.app.level < 27:
                    app.app.level = 27
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level27)
                    if app.app.level < 27:
                        app.app.level = 27
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level27(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 27: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level24)
                if app.app.level < 28:
                    app.app.level = 28
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level28)
                    if app.app.level < 28:
                        app.app.level = 28
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)


class level28(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage , app.keyLetters = homophonicCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 28: Decrypt Homophonic Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def keyOfLetters(app, canvas):
        key = str(app.keyLetters)
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{key[0:100]}\n{key[100:201]}\n{key[201:301]}\n{key[301:401]}\n{key[401:501]}\n{key[500:]}', font = 'Arial 20')
    
    def instructions(app, canvas):
        text = "Using Homophonic Substitution, decrypt the \nbolded message below.The key\nis provided to help you get started.Press 's' to get started"
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level29)
                if app.app.level < 29:
                    app.app.level = 29
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level29)
                    if app.app.level < 29:
                        app.app.level = 29
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.keyOfLetters(canvas)
        app.instructions(canvas)

class level29(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.indexWord = random.randrange(0,8)
        app.secretWord = codeWords[app.indexWord]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = polygramCipher(app.message, app.secretWord)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 29: Decrypt Polygram Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def codeWord(app, canvas):
        word = app.secretWord
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{word}', font = 'Arial 40')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, decrypt the \nbolded message below.The code word\nis provided to help you get started.Press 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level30)
                if app.app.level < 30:
                    app.app.level = 30
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level30)
                    if app.app.level < 30:
                        app.app.level = 30
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.codeWord(canvas)
        app.instructions(canvas)


class level30(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 30: Encrypt Polygram Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, first input a message\nthen input your code word then the encrypted message.\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start\nInput 'x' into all three text boxes to exit"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            userWord = app.getUserInput('Enter your code word')
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptPoly = decodePolygramCipher(encryptedMessage, userWord)
            if (encryptedMessage == decryptPoly):
                app.app.setActiveMode(app.app.level31)
                if app.app.level < 31:
                    app.app.level = 31
                    app.app.lives = 3
            if (encryptedMessage == 'x') or userMessage == 'x' or userWord == 'x':
                app.app.setActiveMode(app.app.levelPage)
            while encryptedMessage != decryptPoly and userMessage != 'x' and userWord != 'x' and encryptedMessage!='x' and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptPoly:
                    app.app.setActiveMode(app.app.level31)
                    if app.app.level < 31:
                        app.app.level = 31
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level31(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.indexWord = random.randrange(0,8)
        app.secretWord = codeWords[app.indexWord]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = polygramCipher(app.message, app.secretWord)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 31: Decrypt Polygram Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def codeWord(app, canvas):
        word = app.secretWord
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{word}', font = 'Arial 40')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, decrypt the \nbolded message below.The code word\nis provided to help you get started.Press 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level32)
                if app.app.level < 32:
                    app.app.level = 32
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level32)
                    if app.app.level < 32:
                        app.app.level = 32
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.codeWord(canvas)
        app.instructions(canvas)

class level32(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 32: Encrypt Polygram Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, first input a message\nthen input your code word then the encrypted message.\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start\nInput 'x' into all three text boxes to exit"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            userWord = app.getUserInput('Enter your code word')
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptPoly = decodePolygramCipher(encryptedMessage, userWord)
            if (encryptedMessage == decryptPoly):
                app.app.setActiveMode(app.app.level33)
                if app.app.level < 33:
                    app.app.level = 33
                    app.app.lives = 3
            if (encryptedMessage == 'x') or userMessage == 'x' or userWord == 'x':
                app.app.setActiveMode(app.app.levelPage)
            while encryptedMessage != decryptPoly and userMessage != 'x' and userWord != 'x' and encryptedMessage!='x' and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptPoly:
                    app.app.setActiveMode(app.app.level33)
                    if app.app.level < 33:
                        app.app.level = 33
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level33(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.indexWord = random.randrange(0,8)
        app.secretWord = codeWords[app.indexWord]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = polygramCipher(app.message, app.secretWord)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 33: Decrypt Polygram Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def codeWord(app, canvas):
        word = app.secretWord
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{word}', font = 'Arial 40')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, decrypt the \nbolded message below.The code word\nis provided to help you get started.Press 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level34)
                if app.app.level < 34:
                    app.app.level = 34
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level34)
                    if app.app.level < 34:
                        app.app.level = 34
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.codeWord(canvas)
        app.instructions(canvas)

class level34(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 34: Encrypt Polygram Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, first input a message\nthen input your code word then the encrypted message.\nIf your encrypted message is properly encoded\nyou win! If not try again! Press 's' to start\nInput 'x' into all three text boxes to exit"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            userWord = app.getUserInput('Enter your code word')
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptPoly = decodePolygramCipher(encryptedMessage, userWord)
            if (encryptedMessage == decryptPoly):
                app.app.setActiveMode(app.app.level35)
                if app.app.level < 35:
                    app.app.level = 35
                    app.app.lives = 3
            if (encryptedMessage == 'x') or userMessage == 'x' or userWord == 'x':
                app.app.setActiveMode(app.app.levelPage)
            while encryptedMessage != decryptPoly and userMessage != 'x' and userWord != 'x' and encryptedMessage!='x' and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptPoly:
                    app.app.setActiveMode(app.app.level35)
                    if app.app.level < 35:
                        app.app.level = 35
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level35(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.indexWord = random.randrange(0,8)
        app.secretWord = codeWords[app.indexWord]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = polygramCipher(app.message, app.secretWord)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 35: Decrypt Polygram Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def codeWord(app, canvas):
        word = app.secretWord
        canvas.create_text(app.width/2, app.height/2 + 100, text = f'{word}', font = 'Arial 40')
    
    def instructions(app, canvas):
        text = "Using Polygram Substitution, decrypt the \nbolded message below.The code word\nis provided to help you get started.Press 's' to start."
        canvas.create_text(app.width/2, app.height/2 - 180, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app,event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level36)
                if app.app.level < 36:
                    app.app.level = 36
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level36)
                    if app.app.level < 36:
                        app.app.level = 36
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.codeWord(canvas)
        app.instructions(canvas)

class level36(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = PlayFairCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 36: Decrypt Play Fair Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Playfair cipher, decrypt the \nbolded message below.The board word\ndoes not contain 'q'. To start press 's'"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level37)
                if app.app.level < 37:
                    app.app.level = 37
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level37)
                    if app.app.level < 37:
                        app.app.level = 37
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level37(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 37: Encrypt Playfair Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using the Playfair, input a message.\nYour board may not contain 'q'.\nIf your encrypted message is properly encoded\nyou win!If not try again! Press 's' to start and 'x'\ntwice in the text box to exit"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptPlay = playFairCipherDecorder(encryptedMessage)
            if (encryptedMessage == decryptPlay):
                app.app.setActiveMode(app.app.level38)
                if app.app.level < 38:
                    app.app.level = 38
                    app.app.lives = 3
            if (encryptedMessage == 'x') or userMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            while encryptedMessage != decryptPlay and userMessage != 'x' and encryptedMessage != 'x' and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptPoly:
                    app.app.setActiveMode(app.app.level38)
                    if app.app.level < 38:
                        app.app.level = 38
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level38(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = PlayFairCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 38: Decrypt Play Fair Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Playfair cipher, decrypt the \nbolded message below.The board word\ndoes not contain 'q'. To start press 's'"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level39)
                if app.app.level < 39:
                    app.app.level = 39
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level39)
                    if app.app.level < 39:
                        app.app.level = 39
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level39(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 39: Encrypt Playfair Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using the Playfair, input a message.\nYour board may not contain 'q'.\nIf your encrypted message is properly encoded\nyou win!If not try again! Press 's' to start and 'x'\ntwice in the text box to exit"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptPlay = playFairCipherDecorder(encryptedMessage)
            if (encryptedMessage == decryptPlay):
                app.app.setActiveMode(app.app.level40)
                if app.app.level < 40:
                    app.app.level = 40
                    app.app.lives = 3
            if (encryptedMessage == 'x') or userMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            while encryptedMessage != decryptPlay and userMessage != 'x' and encryptedMessage != 'x' and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptPoly:
                    app.app.setActiveMode(app.app.level40)
                    if app.app.level < 40:
                        app.app.level = 40
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level40(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = PlayFairCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 40: Decrypt Play Fair Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Playfair cipher, decrypt the \nbolded message below.The board word\ndoes not contain 'q'. To start press 's'"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level41)
                if app.app.level < 41:
                    app.app.level = 41
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level41)
                    if app.app.level < 41:
                        app.app.level = 41
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level41(Mode):
    def appStarted(app):
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 41: Encrypt Playfair Cipher', font = 'Arial 40 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def instructions(app, canvas):
        text = "Using the Playfair, input a message.\nYour board may not contain 'q'.\nIf your encrypted message is properly encoded\nyou win!If not try again! Press 's' to start and 'x'\ntwice in the text box to exit"
        canvas.create_text(app.width/2, app.height/2, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            userMessage = app.getUserInput('Enter your message!')
            encryptedMessage = app.getUserInput('Enter your encrypted message!')
            decryptPlay = playFairCipherDecorder(encryptedMessage)
            if (encryptedMessage == decryptPlay):
                app.app.setActiveMode(app.app.level42)
                if app.app.level < 42:
                    app.app.level = 42
                    app.app.lives = 3
            if (encryptedMessage == 'x') or userMessage == 'x':
                app.app.setActiveMode(app.app.levelPage)
            while encryptedMessage != decryptPlay and userMessage != 'x' and encryptedMessage != 'x' and app.app.lives > 1:
                app.errorMessage
                encryptedMessage = app.getUserInput('Enter your encrypted message!')
                app.app.lives -=1 
                if encryptedMessage == decryptPoly:
                    app.app.setActiveMode(app.app.level42)
                    if app.app.level < 42:
                        app.app.level = 42
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)

class level42(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = PlayFairCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 42: Decrypt Play Fair Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Playfair cipher, decrypt the \nbolded message below.The board word\ndoes not contain 'q'. To start press 's'"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level43)
                if app.app.level < 43:
                    app.app.level = 43
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level43)
                    if app.app.level < 43:
                        app.app.level = 43
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level43(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = hillCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 43: Decrypt Hill Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Hill cipher, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level44)
                if app.app.level < 44:
                    app.app.level = 44
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level44)
                    if app.app.level < 44:
                        app.app.level = 44
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level44(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = hillCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 44: Decrypt Hill Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Hill cipher, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level45)
                if app.app.level < 45:
                    app.app.level = 45
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level45)
                    if app.app.level < 45:
                        app.app.level = 45
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level45(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = hillCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 45: Decrypt Hill Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Hill cipher, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level46)
                if app.app.level < 46:
                    app.app.level = 46
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level46)
                    if app.app.level < 46:
                        app.app.level = 46
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level46(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = hillCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 46: Decrypt Hill Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Hill cipher, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level47)
                if app.app.level < 47:
                    app.app.level = 47
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level47)
                    if app.app.level < 47:
                        app.app.level = 47
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level47(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = hillCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 47: Decrypt Hill Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Hill cipher, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level48)
                if app.app.level < 48:
                    app.app.level = 48
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level48)
                    if app.app.level < 48:
                        app.app.level = 48
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

class level48(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.encryptedMessage = hillCipher(app.message)
        app.level = app.app.level
        app.lives = app.app.lives
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 48: Decrypt Hill Cipher', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')
    
    def enocodedMessage(app,canvas):
        if len(app.message) <= 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 45 bold')
        if len(app.message) > 30:
            canvas.create_text(app.width/2, app.height/2, text = f'{app.encryptedMessage}', font = 'Arial 30 bold')
    
    def instructions(app, canvas):
        text = "Using the Hill cipher, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            phrase = app.getUserInput('Try to decode the message!')
            if (phrase == app.message):
                app.app.setActiveMode(app.app.level49)
                if app.app.level < 49:
                    app.app.level = 49
                    app.app.lives = 3
            if (phrase == None):
                app.app.setActiveMode(app.app.levelPage)
            while phrase != app.message and phrase != None and app.app.lives > 1:
                app.errorMessage
                phrase = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if phrase == app.message:
                    app.app.setActiveMode(app.app.level49)
                    if app.app.level < 49:
                        app.app.level = 49
                        app.app.lives = 3
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)
    
    def redrawAll(app,canvas):
        app.title(canvas)
        app.enocodedMessage(canvas)
        app.instructions(canvas)

# from 15112 notes 
# https://www.cs.cmu.edu/~112/notes/notes-loops.html#isPrime
def isPrime(n):
    if (n < 2):
        return False
    if (n == 2):
        return True
    if (n % 2 == 0):
        return False
    maxFactor = round(n**0.5)
    for factor in range(3,maxFactor+1,2):
        if (n % factor == 0):
            return False
    return True

# the following is the RSA algorithm which I used to main sources for 
# https://sites.psu.edu/gottiparthyanirudh/writing-sample-3/ and 
# https://brilliant.org/wiki/rsa-encryption/
# alot of the modular arithmetic I used code for but other than that I wrote it
# the first three functions are taken from the first source and the rest, unless 
# highlighted otherwise is from me. 

# finding a relative prime (e)
# the following three functions came from 
# https://sites.psu.edu/gottiparthyanirudh/writing-sample-3/
#Euclid's Algorithm
def eugcd(e,r):
    for i in range(1,r):
        while(e!=0):
            a,b=r//e,r%e
            r=e
            e=b
            return e 
 
#Extended Euclidean Algorithm
def eea(a,b):
    if(a%b==0):
        return(b,0,1)
    else:
        gcd,s,t = eea(b,a%b)
        s = s-((a//b) * t)
        return(gcd,t,s)
 
#Multiplicative Inverse
def mult_inv(e,r):
    gcd,s,_=eea(e,r)
    if(gcd!=1):
        return None
    else:
        return s%r

# similar to https://sites.psu.edu/gottiparthyanirudh/writing-sample-3/
def RSAencryption(publicKey, message):
    e, n = publicKey 
    m = 0 
    x = []
    message = message.upper()
    message = message.replace(" ", '')
    for i in range(len(message)):
        char = message[i]
        m = ord(char) - 65 
        cipherNum = (m**e) % n 
        x.append(cipherNum)
    return x

def RSAdecryption(privateaKey, encryptedMessage):
    d, n = privateKey 
    result = ''
    m = 0
    encryptedMessage = encryptedMessage.replace(' ', '')
    for i in range(len(encryptedMessage)):
        num = encryptedMessage[i]
        m = (num ** d) % n
        m += 65
        result += chr(m)
    return result 

class level49(Mode):
    def appStarted(app):
        app.index = random.randrange(0, 34)
        app.message = sentences[app.index]
        app.errorMessage = 'Try Again'
        app.level = app.app.level
        app.lives = app.app.lives
        app.numP = 0
        app.numQ = 0
        app.eut = 0 
        app.egcd = 0
        app.e = 0
        app.public = (0, 0)
        app.private = (0, 0)
        app.displayP = False 
        app.encryptedMessage = ''
    
    def title(app,canvas):
        canvas.create_text(app.width/2, 100, text = 'Level 49: Decrypt RSA', font = 'Arial 38 bold')
        canvas.create_rectangle(app.width/2 - 100, 600, app.width/2 + 100, 700, fill = 'purple')
        canvas.create_text(app.width/2, 650, text = 'Exit Level', font = 'Times 25 bold', fill = 'white')

    
    def instructions(app, canvas):
        text = "Using the RSA cryptosystem, decrypt the \nbolded message below. Press 's' to start!"
        canvas.create_text(app.width/2, app.height/2 - 170, text = f'{text}', font = 'Arial 30', fill = 'Red')

    def mousePressed(app, event):
        if app.width/2 - 100 <= event.x <= app.width/2 + 100 and 600 <= event.y <= 700:
            app.app.setActiveMode(app.app.levelPage)

    def keyPressed(app, event):
        if event.key == 's':
            app.numP = int(app.getUserInput('Enter a prime number for p'))
            app.numQ = int(app.getUserInput('Enter a prime number for q'))
            check_P = isPrime(app.numP)
            check_Q = isPrime(app.numQ)
            while check_P == False or check_Q == False:
                app.numP = int(app.getUserInput('Enter a prime number for p'))
                app.numQ = int(app.getUserInput('Enter a prime number for q'))
                check_P = isPrime(app.numP)
                check_Q = isPrime(app.numQ)
            # the math for the next three come from https://brilliant.org/wiki/rsa-encryption/
            # next porition is to get n = pq this is the first half of the public key
            n = app.numP * app.numQ
            # next is to get eulers toitent 
            app.eut = (app.numP - 1) * (app.numQ - 1)
            # next find e which is relatively prime to app.eut so we need to find an e thats gcd to app.eut is 1:
            app.egcd = eugcd(3, app.eut)
            # we know need to find the largest value of e 
            for i in range(1, 1000):
                if eugcd(i, app.eut) == 1:
                    app.e = i 
            # calculate the private key
            d = mult_inv(app.e, app.eut)
            app.public = (app.e, n)
            app.private = (d, n)
            app.displayP = True 
            app.encryptedMessage = RSAencryption(app.public, app.message)
            app.answer = app.getUserInput('Enter the decrypted message')
            if app.answer == app.message:
                app.app.setActiveMode(app.app.celebrationPage)
            if app.answer == None or app.numP == None or app.numQ == None:
                app.app.setActiveMode(app.app.levelPage)
            while app.answer != app.message and app.answer!= None and app.app.lives > 1:
                app.errorMessage
                app.answer = app.getUserInput('Try to decode the message!')
                app.app.lives -=1 
                if app.answer == app.message:
                    app.app.setActiveMode(app.app.celebrationPage)
            if app.app.lives == 1:
                app.app.setActiveMode(app.app.gameOver)

        
    def displayPrivate(app,canvas):
        if app.displayP == True:
            canvas.create_text(app.width/2, app.height/2 + 100, text = f'The private key is {app.private} ', font = 'Times 30 bold', fill = 'red')
            if len(app.encryptedMessage) > 30:
                canvas.create_text(app.width/2, app.height/2 + 70, text = f'The ecrypted message is {app.encryptedMessage}', font = 'Times 18 bold', fill = 'dark blue')
            if len(app.encryptedMessage) < 30:
                canvas.create_text(app.width/2, app.height/2 + 70, text = f'The ecrypted message is {app.encryptedMessage}', font = 'Times 25 bold', fill = 'dark blue')

    def redrawAll(app,canvas):
        app.title(canvas)
        app.instructions(canvas)
        app.displayPrivate(canvas)

class celebrationPage(Mode):
    def redrawAll(app,canvas):
        canvas.create_text(app.width/2, app.height/2, text= 'Congrats your an expert at\nencryption and decryption', fill = 'purple', font = 'Times 40 bold')

class practiceMode(Mode):
    def title(app,canvas):
        canvas.create_text(app.width/2, 40, text = 'Practice Mode', font = 'Times 50 bold', fill = 'blue')

    def practiceSelection(app, canvas):
        canvas.create_rectangle(app.width/4 - app.width/8, 100, app.width/8 + app.width/4 , 190, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/8, 300, app.width/8 + app.width/4 , 390, fill = 'red')
        canvas.create_rectangle(app.width/4 - app.width/8, 500, app.width/8 + app.width/4 , 590, fill = 'red')
        canvas.create_rectangle(app.width/2 + app.width/8, 500, app.width/2 + app.width/2.7 , 590, fill = 'red')
        canvas.create_rectangle(app.width/2 + app.width/8, 300, app.width/2 + app.width/2.7 , 390, fill = 'purple')
        canvas.create_rectangle(app.width/2 + app.width/8, 100, app.width/2 + app.width/2.7 , 190, fill = 'red')
        canvas.create_text(app.width/2 + app.width/4, 345, text = '     Play\ndeCryptii', fill = 'white', font = 'Times 25 bold')
        canvas.create_text(app.width/8 + app.width/4 - 100, 140, text = 'Caesar Cipher', font = 'Times 25 bold', fill = 'white')
        canvas.create_text(app.width/8 + app.width/4 - 100, 345, text = 'Polygram', font = 'Times 25 bold', fill = 'white')
        canvas.create_text(app.width/8 + app.width/4 - 100, 545, text = 'Monoalphabetic', font = 'Times 25 bold', fill = 'white')
        canvas.create_text(app.width/2 + app.width/4, 140, text = 'Playfair', font = 'Times 25 bold', fill = 'white')
        canvas.create_text(app.width/2 + app.width/4, 545, text = 'Hill', font = 'Times 25 bold', fill = 'white')
        canvas.create_text(app.width/2, app.height/2 + 45, text = "Click on any of the red boxes to practice", font = 'Times 30 bold', fill = 'blue')
    
    def mousePressed(app,event):
        if ((app.width/4 - app.width/8) <= event.x <= (app.width/8 + app.width/4)) and (100 <= event.y <= 190):
            app.app.setActiveMode(app.app.caesarPractice)
        if ((app.width/4 - app.width/8) <= event.x <= (app.width/8 + app.width/4)) and (300 <= event.y <= 390):
            app.app.setActiveMode(app.app.polygramPractice)
        if ((app.width/4 - app.width/8) <= event.x <= (app.width/8 + app.width/4)) and (500 <= event.y <= 590):
            app.app.setActiveMode(app.app.monoalphabeticPractice)
        if ((app.width/2 + app.width/8) <= event.x <= (app.width/2 + app.width/2.7)) and (500 <= event.y <= 590):
            app.app.setActiveMode(app.app.hillPractice)
        if ((app.width/2 + app.width/8) <= event.x <= (app.width/2 + app.width/2.7)) and (300 <= event.y <= 390):
            app.app.setActiveMode(app.app.levelPage)
        if ((app.width/2 + app.width/8) <= event.x <= (app.width/2 + app.width/2.7)) and (100 <= event.y <= 190):
            app.app.setActiveMode(app.app.playfairPractice)
    
    def keyPressed(app,event):
        if event.key == 'x':
            app.app.setActiveMode(app.app.instructionPage)

    def redrawAll(app,canvas):
        app.title(canvas)
        app.practiceSelection(canvas)

class caesarPractice(Mode):
    def appStarted(app):
        app.message = ''
        app.shift = 0
        app.displayMessage = False 
        app.count = 0
        app.displayIndex = False
        app.lAndI = dict()
        app.displayShift = False 
        app.shiftDict = dict()
        app.displaySD = False
        app.final = False 
        app.decode = False 
        app.visualize = True
        app.practice = False 
        app.goodJob = False 

    def title(app, canvas):
        if app.decode == True and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Caesar Cipher Practice', font = 'Times 50 bold')
            instructionText = "To decode the caesar cipher, shift each letter back n times.\nN is the shift number which has been provided.\nIf you shift and get to 'a' and still have more shift spaces,\nwrap around back to z and continue the pattern."
            canvas.create_text(app.width/2, 200, fill = 'black', text = f'{instructionText}', font = 'Times 30')
            canvas.create_text(app.width/2, 300, fill = 'black', text = 'Example', font = 'Times 35 bold underline')
        if app.decode == False and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Caesar Cipher Practice', font = 'Times 50 bold')
            instructionText = "To encode the caesar cipher, shift each letter forward n times.\nN is the shift number which has been provided.\nIf you shift and get to 'z' and still have more shift spaces,\nwrap around back to 'a' and continue the pattern."
            canvas.create_text(app.width/2, 200, fill = 'black', text = f'{instructionText}', font = 'Times 30')
            canvas.create_text(app.width/2, 300, fill = 'black', text = 'Example', font = 'Times 35 bold underline')


    def keyPressed(app,event):
        if event.key == 'd':
            app.message = ''
            app.shift = 0
            app.displayMessage = False 
            app.count = 0
            app.displayIndex = False
            app.lAndI = dict()
            app.displayShift = False 
            app.shiftDict = dict()
            app.displaySD = False
            app.final = False 
            app.decode = True 
            app.goodJob = False 
            app.visualize = True
        if event.key == 's':
            app.message = ''
            app.shift = 0
            app.displayMessage = False 
            app.count = 0
            app.displayIndex = False
            app.lAndI = dict()
            app.displayShift = False 
            app.shiftDict = dict()
            app.displaySD = False
            app.final = False 
            app.decode = False
            app.visualize = False
            app.goodJob = False 
            app.practice = True 

    def mousePressed(app, event):
        if event.x and app.count == 0  and app.practice == False:
            if app.decode == False:
                app.message = app.getUserInput('enter a word')
                app.shift = app.getUserInput('enter a whole number less than 25')
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                    app.message = ''
                    app.shift = 0
                    app.displayMessage = False 
                    app.count = 0
                    app.displayIndex = False
                    app.lAndI = dict()
                    app.displayShift = False 
                    app.shiftDict = dict()
                    app.displaySD = False
                    app.final = False 
                    app.decode = False 
                    app.visualize = True
                    app.practice = False 
                    app.goodJob = False
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                    app.message = ''
                    app.shift = 0
                    app.displayMessage = False 
                    app.count = 0
                    app.displayIndex = False
                    app.lAndI = dict()
                    app.displayShift = False 
                    app.shiftDict = dict()
                    app.displaySD = False
                    app.final = False 
                    app.decode = False 
                    app.visualize = True
                    app.practice = False 
                    app.goodJob = False
                app.shift = int(app.shift)
                app.displayMessage = True 
                app.count += 1
            if app.decode == True:
                app.message = app.getUserInput('enter an ecrypted word')
                app.shift = app.getUserInput('enter the shift')
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                app.shift = int(app.shift)
                app.displayMessage = True 
                app.count += 1
        if event.x and app.count == 1 and app.practice == False:
            app.displayIndex = True 
            app.count += 1 
        if event.x and app.count >= 1 and app.practice == False:
            app.displayShift = True
        if event.x and app.count >= 1 and app.practice == False:
            app.displaySD = True 
        if event.x and app.count >= 1 and app.practice == False:
            app.final = True 
        if app.practice == True:
            if app.width/2 - 350 <= event.x <= app.width/2 - 150 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                shift = random.randrange(0,24)
                userEncode = app.getUserInput(f'Enter encoded message for {message} with a shift of {shift} ')
                if userEncode == applyCaesarCipher(message, shift):
                    app.goodJob = True 
                while userEncode != applyCaesarCipher(message, shift) and userEncode != None:
                    userEncode = app.getUserInput(f'Try again! Enter encoded message for {message} with a shift of {shift} ')
                    if userEncode == applyCaesarCipher(message, shift):
                        app.goodJob = True 
                if userEncode == None:
                    pass
            if app.width/2 + 150 <= event.x <=  app.width/2 + 350 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                app.goodJob = False 
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                shift = random.randrange(0,24)
                encryptedMessage = applyCaesarCipher(message, shift)
                userDecode = app.getUserInput(f'Decrypt {encryptedMessage} with a shift of {shift}')
                if userDecode == message:
                    app.goodJob = True 
                while userDecode != message and userDecode != None:
                    userEncode = app.getUserInput(f'Try again! Decode {encryptedMessage} with a shift of {shift} ')
                    if uuserDecode == message:
                        app.goodJob = True 
                if userDecode == None:
                    pass
            if app.width/2 - 100 <= event.x <= app.width/2 + 100 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.goodJob = False
                app.app.setActiveMode(app.app.practiceMode)
            if app.width/2 - 350 <= event.x <= app.width/2 - 200 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = True 
                app.goodJob = False 
                app.visualize = True
                app.practice = False
            if app.width/2 + 200 <= event.x <= app.width/2 + 350 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = False 
                app.goodJob = False 
                app.visualize = True
                app.practice = False

    def displayM(app,canvas):
        if app.displayMessage == True:
            canvas.create_text(app.width/2, 350, text = f'Plain Text: {app.message}\tShift:{app.shift}', font = 'Times 30 bold', fill = 'dark blue')
    
    def displayI(app,canvas):
        if app.displayIndex == True:
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            app.lAndI = dict()
            for i in range(len(app.message)):
                letter = app.message[i]
                index = alphabet.find(letter)
                app.lAndI[letter] = index 
            canvas.create_text(app.width/2, 380, text = 'Find the index of every letter of the message in the alphabet', fill = 'black', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 415, text = f'{app.lAndI}', fill = 'red', font = 'Times 25 bold')
    
    def displayS(app, canvas):
        if app.displayShift == True and app.decode == False:
            for i in range(len(app.message)):
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                letter = app.message[i]
                index = alphabet.find(letter)
                app.shiftDict[letter] = (index + app.shift) % 26
            canvas.create_text(app.width/2, 440, text = 'Calculate the shifted value by adding the shift to the index value of each letter', fill = 'black', font = 'Times 25 bold')
            canvas.create_text(app.width/2, 475, text = f'{app.shiftDict}', fill = 'red', font = 'Times 25 bold')
        if app.displayShift == True and app.decode == True:
            for i in range(len(app.message)):
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                letter = app.message[i]
                index = alphabet.find(letter)
                app.shiftDict[letter] = abs((index - app.shift) % 26)
            canvas.create_text(app.width/2, 440, text = 'Calculate the original index by subtracting the shift from the index of the letter', fill = 'black', font = 'Times 25 bold')
            canvas.create_text(app.width/2, 475, text = f'{app.shiftDict}', fill = 'red', font = 'Times 25 bold')

    def displayDictShifted(app,canvas):
        if app.displaySD == True and app.decode == False:
            shiftedLetters = dict()
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            for i in range(len(app.message)):
                letter = app.message[i]
                index = alphabet.find(letter)
                shift = (index + app.shift) % 26
                shiftedLetters[letter] = alphabet[shift]
            canvas.create_text(app.width/2, 515, text = 'Find the letter by using the new index, the original letter is on the left and\n\tencrypted letter is on the right of the colon', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 575, text = f'{shiftedLetters}', font = 'Times 30 bold', fill = 'red')
    
    def displayFinal(app,canvas):
        if app.final == True and app.decode == False:
            result = applyCaesarCipher(app.message, app.shift)
            canvas.create_text(app.width/2, 600, text = 'Compile all the enrypted letters in the same position as the original letters', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 640, text = f'{result}', fill = 'red', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 680, text = "Press 'd' to visualize decoding Caesar's Cipher", fill = 'purple', font = 'Times 30 bold')
        if app.final == True and app.decode == True:
            result = ''
            for i in range(len(app.message)):
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                letter = app.message[i]
                index = alphabet.find(letter)
                result += alphabet[abs((index - app.shift) % 26)]
            canvas.create_text(app.width/2, 520, text = 'Compile all the enrypted letters in the same position as the original letters', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 550, text = f'{result}', fill = 'red', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 590, text = "Press 's' to practice decoding Caesar's Cipher", fill = 'purple', font = 'Times 30 bold')

    def instructionPractice(app,canvas):
        if app.practice == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Caesar Cipher Practice', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 165, fill = 'black', text = 'Press on the green button to practice encoding, blue for decoding,\n\tand the purple to exit the practice', font = 'Times 30')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 - 100, app.width/2 - 150, app.height/2 + 100, fill = 'green')
            canvas.create_rectangle(app.width/2 + 150, app.height/2 - 100, app.width/2 + 350, app.height/2 + 100, fill = 'light blue')
            canvas.create_rectangle(app.width/2 - 100, app.height/2 + 150, app.width/2 + 100, app.height/2 + 300, fill = 'purple')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 + 150, app.width/2 - 200, app.height/2 + 300, fill = 'pink')
            canvas.create_rectangle(app.width/2 + 200, app.height/2 + 150, app.width/2 + 350, app.height/2 + 300, fill = 'orange')
            canvas.create_text(app.width/2 - 250, app.height/2, text = 'Practice\nEncoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 + 250, app.height/2, text = 'Practice\nDecoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2, app.height/2 + 225, text = 'Exit', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 - 275, app.height/2 + 225, text = 'Visualize\nDecoding', font = 'Times 25 bold', fill = 'black')
            canvas.create_text(app.width/2 + 275, app.height/2 + 225, text = 'Visualize\nEncoding', font = 'Times 25 bold', fill = 'black')

    
    def celebration(app,canvas):
        if app.goodJob == True:
            canvas.create_text(app.width/2, app.height/2, text = 'Congrats!', font = 'Times 30 bold', fill = 'pink')

    def redrawAll(app, canvas):
        app.title(canvas)
        app.displayM(canvas)
        app.displayI(canvas)
        app.displayS(canvas)
        app.displayDictShifted(canvas)
        app.displayFinal(canvas)
        app.instructionPractice(canvas)
        app.celebration(canvas)

class polygramPractice(Mode):
    def appStarted(app):
        app.message = ''
        app.shift = 0
        app.displayMessage = False 
        app.count = 0
        app.displayIndex = False
        app.lAndI = dict()
        app.displayShift = False 
        app.shiftDict = dict()
        app.displaySD = False
        app.final = False 
        app.decode = False 
        app.visualize = True
        app.practice = False 
        app.goodJob = False 

    def title(app, canvas):
        if app.decode == True and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Polygram Subsititution Practice', font = 'Times 50 bold')
            instructionText = "        To decode a polygram substitution cipher, assign each\n     letter of the code word to a letter in the encrypted message.\n\t          For each pair, calculate the shift.\n\t    Subtract this shift from the orignial letter,\n\t   the index of the letter in the alphabet.\nUsing this new number, use that as the new index of the alphabet."
            canvas.create_text(app.width/2, 240, fill = 'black', text = f'{instructionText}', font = 'Times 30')
            canvas.create_text(app.width/2, 360, fill = 'black', text = 'Example', font = 'Times 35 bold underline')
        if app.decode == False and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Polygram Subsititution Practice', font = 'Times 50 bold')
            instructionText = "        To encode a polygram substitution cipher, assign each\n     letter of the code word to a letter in the message.\n\t          For each pair, calculate the shift.\n\t    Add this shift to the orignial letter,\n\t   the index of the letter in the alphabet.\nUsing this new number, use that as the new index of the alphabet."
            canvas.create_text(app.width/2, 240, fill = 'black', text = f'{instructionText}', font = 'Times 30')
            canvas.create_text(app.width/2, 360, fill = 'black', text = 'Example', font = 'Times 35 bold underline')


    def keyPressed(app,event):
        if event.key == 'd':
            app.message = ''
            app.shift = 0
            app.displayMessage = False 
            app.count = 0
            app.displayIndex = False
            app.lAndI = dict()
            app.displayShift = False 
            app.shiftDict = dict()
            app.displaySD = False
            app.final = False 
            app.decode = True 
            app.goodJob = False 
            app.visualize = True
        if event.key == 's':
            app.message = ''
            app.shift = 0
            app.displayMessage = False 
            app.count = 0
            app.displayIndex = False
            app.lAndI = dict()
            app.displayShift = False 
            app.shiftDict = dict()
            app.displaySD = False
            app.final = False 
            app.decode = False
            app.visualize = False
            app.goodJob = False 
            app.practice = True 

    def mousePressed(app, event):
        if event.x and app.count == 0  and app.practice == False:
            if app.decode == False:
                app.message = app.getUserInput('enter a word')
                app.shift = app.getUserInput('enter a code')
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                    app.message = ''
                    app.shift = 0
                    app.displayMessage = False 
                    app.count = 0
                    app.displayIndex = False
                    app.lAndI = dict()
                    app.displayShift = False 
                    app.shiftDict = dict()
                    app.displaySD = False
                    app.final = False 
                    app.decode = False 
                    app.visualize = True
                    app.practice = False 
                    app.goodJob = False
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                    app.message = ''
                    app.shift = 0
                    app.displayMessage = False 
                    app.count = 0
                    app.displayIndex = False
                    app.lAndI = dict()
                    app.displayShift = False 
                    app.shiftDict = dict()
                    app.displaySD = False
                    app.final = False 
                    app.decode = False 
                    app.visualize = True
                    app.practice = False 
                    app.goodJob = False
                app.displayMessage = True 
                app.count += 1
            if app.decode == True:
                app.message = app.getUserInput('enter an ecrypted word')
                app.shift = app.getUserInput('enter the code word')
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                app.displayMessage = True 
                app.count += 1
        if event.x and app.count == 1 and app.practice == False:
            app.displayIndex = True 
            app.count += 1 
        if event.x and app.count >= 1 and app.practice == False:
            app.displayShift = True
        if event.x and app.count >= 1 and app.practice == False:
            app.displaySD = True 
        if event.x and app.count >= 1 and app.practice == False:
            app.final = True 
        if app.practice == True:
            if app.width/2 - 350 <= event.x <= app.width/2 - 150 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                shift = random.randrange(0,7)
                codeWord = codeWords[shift]
                userEncode = app.getUserInput(f'Enter encoded message for {message} with a code word of {codeWord} ')
                if userEncode == polygramCipher(message, codeWord):
                    app.goodJob = True 
                while userEncode != polygramCipher(message, codeWord) and userEncode != None:
                    userEncode = app.getUserInput(f'Try again! Enter encoded message for {message} with a code word of {codeWord} ')
                    if userEncode == polygramCipher(message, codeWord):
                        app.goodJob = True 
                if userEncode == None:
                    pass
            if app.width/2 + 150 <= event.x <=  app.width/2 + 350 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                app.goodJob = False 
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                shift = random.randrange(0,7)
                codeWord = codeWords[shift]
                encryptedMessage = polygramCipher(message, codeWord)
                userDecode = app.getUserInput(f'Decrypt {encryptedMessage} with a code word of {codeWord}')
                if userDecode == message:
                    app.goodJob = True 
                while userDecode != message and userDecode != None:
                    userEncode = app.getUserInput(f'Try again! Decrypt {encryptedMessage} with a code word of {codeWord}')
                    if uuserDecode == message:
                        app.goodJob = True 
                if userDecode == None:
                    pass
            if app.width/2 - 100 <= event.x <= app.width/2 + 100 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.goodJob = False
                app.app.setActiveMode(app.app.practiceMode)
            if app.width/2 - 350 <= event.x <= app.width/2 - 200 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = True 
                app.goodJob = False 
                app.visualize = True
                app.practice = False
            if app.width/2 + 200 <= event.x <= app.width/2 + 350 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = False 
                app.goodJob = False 
                app.visualize = True
                app.practice = False

    def displayM(app,canvas):
        if app.displayMessage == True:
            canvas.create_text(app.width/2, 400, text = f'Plain Text: {app.message}\tCode word:{app.shift}', font = 'Times 30 bold', fill = 'dark blue')
    
    def displayI(app,canvas):
        if app.displayIndex == True:
            corresponding = codeWordShift(app.message, app.shift)
            rDict = []
            for i in range(len(app.message)):
                charOC = corresponding[i]
                messageChar = app.message[i]
                rDict.append([messageChar,charOC]) 
            canvas.create_text(app.width/2, 430, text = 'Pair each letter of the message to one of the code word', fill = 'black', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 465, text = f'{rDict}', fill = 'red', font = 'Times 30 bold')
    
    def displayS(app, canvas):
        if app.displayShift == True:
            corresponding = codeWordShift(app.message, app.shift)
            shiftDict = []
            for i in range(len(app.message)):
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                letter = app.message[i]
                indexM = alphabet.find(letter)
                letterC = corresponding[i]
                indexC = alphabet.find(letterC)
                fShift = (indexM + indexC) % 26 
                shiftDict.append([letter, fShift])
            canvas.create_text(app.width/2, 495, text = 'Calculate the shift value of each letter', fill = 'black', font = 'Times 25 bold')
            canvas.create_text(app.width/2, 525, text = f'{shiftDict}', fill = 'red', font = 'Times 25 bold')
    
    def displayFinal(app,canvas):
        if app.final == True and app.decode == False:
            result = polygramCipher(app.message, app.shift)
            canvas.create_text(app.width/2, 580, text = '\t        Add the shift to the index of the original letter.\nThen compile all the enrypted letters in the same position as the original letters', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 640, text = f'Result : {result}', fill = 'red', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 680, text = "Press 'd' to visualize decoding Polygram Substitution Cipher", fill = 'purple', font = 'Times 30 bold')
        if app.final == True and app.decode == True:
            result = ''
            cores = codeWordShift(app.message, app.shift)
            for i in range(len(app.message)):
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                letter = app.message[i]
                index = alphabet.find(letter)
                letterC = cores[i]
                indexC = alphabet.find(letterC)
                shift = abs(index - indexC) % 26
                result += alphabet[shift]
            canvas.create_text(app.width/2, 580, text = '\t        Add the shift to the index of the original letter.\nThen compile all the enrypted letters in the same position as the original letters', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 660, text = f'Result : {result}', fill = 'red', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 700, text = "Press 's' to practice decoding and encoding Polygram Substitution Cipher", fill = 'purple', font = 'Times 30 bold')
            

    def instructionPractice(app,canvas):
        if app.practice == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Polygram Substitution Practice', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 165, fill = 'black', text = 'Press on the green button to practice encoding, blue for decoding,\n\tand the purple to exit the practice', font = 'Times 30')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 - 100, app.width/2 - 150, app.height/2 + 100, fill = 'green')
            canvas.create_rectangle(app.width/2 + 150, app.height/2 - 100, app.width/2 + 350, app.height/2 + 100, fill = 'light blue')
            canvas.create_rectangle(app.width/2 - 100, app.height/2 + 150, app.width/2 + 100, app.height/2 + 300, fill = 'purple')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 + 150, app.width/2 - 200, app.height/2 + 300, fill = 'pink')
            canvas.create_rectangle(app.width/2 + 200, app.height/2 + 150, app.width/2 + 350, app.height/2 + 300, fill = 'orange')
            canvas.create_text(app.width/2 - 250, app.height/2, text = 'Practice\nEncoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 + 250, app.height/2, text = 'Practice\nDecoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2, app.height/2 + 225, text = 'Exit', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 - 275, app.height/2 + 225, text = 'Visualize\nDecoding', font = 'Times 25 bold', fill = 'black')
            canvas.create_text(app.width/2 + 275, app.height/2 + 225, text = 'Visualize\nEncoding', font = 'Times 25 bold', fill = 'black')

    
    def celebration(app,canvas):
        if app.goodJob == True:
            canvas.create_text(app.width/2, app.height/2, text = 'Congrats!', font = 'Times 30 bold', fill = 'pink')
    
    def redrawAll(app, canvas):
        app.title(canvas)
        app.displayM(canvas)
        app.displayI(canvas)
        app.displayS(canvas)
        app.displayFinal(canvas)
        app.instructionPractice(canvas)
        app.celebration(canvas)
def monoAlphabeticCipherWithKey(word, key):
    result = "" 

    # follows the same pattern as ROT13 but with the unique key that was generated 
    for char in range(len(word)):
        letter = word[char]
        nL = letter.lower()
        if nL in key:
            if letter.isupper() == False:
                keyIndex = key[letter]
                result += keyIndex
            else:
                lower = letter.lower()
                kIndex = key[lower]
                result += kIndex
        else:
            result += letter
    return result
class monoalphabeticPractice(Mode):  
    def appStarted(app):
        app.message = ''
        app.key = generateMonoalphabeticCipherKey()
        app.displayMessage = False 
        app.count = 0
        app.displayIndex = False
        app.lAndI = dict()
        app.displayShift = False 
        app.shiftDict = dict()
        app.displaySD = False
        app.final = False 
        app.decode = False 
        app.visualize = True
        app.practice = False 
        app.goodJob = False 

    def title(app, canvas):
        if app.decode == True and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Monoalphabetic Substitution Practice', font = 'Times 40 bold')
            instructionText = "To decode a monoalphabetic substitution, each letter is encrypted\n      with another letter so for example every A that would be\n\tin the orignal text would be encrypted to a D.\n             Using the key given, find each letter and swap it."
            canvas.create_text(app.width/2, 200, fill = 'black', text = f'{instructionText}', font = 'Times 30')
            canvas.create_text(app.width/2, 300, fill = 'black', text = 'Example', font = 'Times 35 bold underline')
        if app.decode == False and app.visualize == True:
            canvas.create_text(app.width/2, 80, fill = 'black', text = 'Monoalphabetic Substitution Practice', font = 'Times 40 bold')
            instructionText = "To encode a monoalphabetic substitution, each letter is encrypted\n  with another letter from a generated key made by the encryptor"
            canvas.create_text(app.width/2, 130, fill = 'black', text = f'{instructionText}', font = 'Times 25')
            canvas.create_text(app.width/2, 180, fill = 'black', text = 'Example', font = 'Times 35 bold underline')


    def keyPressed(app,event):
        if event.key == 'd':
            app.message = ''
            app.shift = 0
            app.displayMessage = False 
            app.count = 0
            app.displayIndex = False
            app.lAndI = dict()
            app.displayShift = False 
            app.shiftDict = dict()
            app.displaySD = False
            app.final = False 
            app.decode = True 
            app.goodJob = False 
            app.visualize = True
        if event.key == 's':
            app.message = ''
            app.shift = 0
            app.displayMessage = False 
            app.count = 0
            app.displayIndex = False
            app.lAndI = dict()
            app.displayShift = False 
            app.shiftDict = dict()
            app.displaySD = False
            app.final = False 
            app.decode = False
            app.visualize = False
            app.goodJob = False 
            app.practice = True 

    def mousePressed(app, event):
        if event.x and app.count == 0  and app.practice == False:
            if app.decode == False:
                app.message = app.getUserInput('enter a word')
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                    app.message = ''
                    app.shift = 0
                    app.displayMessage = False 
                    app.count = 0
                    app.displayIndex = False
                    app.lAndI = dict()
                    app.displayShift = False 
                    app.shiftDict = dict()
                    app.displaySD = False
                    app.final = False 
                    app.decode = False 
                    app.visualize = True
                    app.practice = False 
                    app.goodJob = False
                app.displayMessage = True 
                app.count += 1
            if app.decode == True:
                app.message = app.getUserInput('enter an ecrypted word')
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                app.displayMessage = True 
                app.count += 1
        if event.x and app.count == 1 and app.practice == False:
            app.displayIndex = True 
            app.count += 1 
        if event.x and app.count >= 1 and app.practice == False:
            app.displayShift = True
        if event.x and app.count >= 1 and app.practice == False:
            app.displaySD = True 
        if event.x and app.count >= 1 and app.practice == False:
            app.final = True 
        if app.practice == True:
            if app.width/2 - 350 <= event.x <= app.width/2 - 150 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                app.key = generateMonoalphabeticCipherKey()
                userEncode = app.getUserInput(f'Enter encoded message for {message} with a key dictionary of {app.key}')
                if userEncode == monoAlphabeticCipherWithKey(message, app.key):
                    app.goodJob = True 
                while userEncode != monoAlphabeticCipherWithKey(message, app.key) and userEncode != None:
                    userEncode = app.getUserInput(f'Try again! Enter encoded message for {message} with a key dictionary of {app.key}')
                    if userEncode == monoAlphabeticCipherWithKey(message, app.key):
                        app.goodJob = True 
                if userEncode == None:
                    pass
            if app.width/2 + 150 <= event.x <=  app.width/2 + 350 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                app.goodJob = False 
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                app.key = generateMonoalphabeticCipherKey()
                encryptedMessage = monoAlphabeticCipherWithKey(message, app.key)
                userDecode = app.getUserInput(f'Decrypt {encryptedMessage} with a key dictionary of {app.key}')
                if userDecode == message:
                    app.goodJob = True 
                while userDecode != message and userDecode != None:
                    userEncode = app.getUserInput(f'Try again! Decrypt {encryptedMessage} with a key of {app.key}')
                    if userDecode == message:
                        app.goodJob = True 
                if userDecode == None:
                    pass
            if app.width/2 - 100 <= event.x <= app.width/2 + 100 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.goodJob = False
                app.app.setActiveMode(app.app.practiceMode)
            if app.width/2 - 350 <= event.x <= app.width/2 - 200 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = True 
                app.goodJob = False 
                app.visualize = True
                app.practice = False
            if app.width/2 + 200 <= event.x <= app.width/2 + 350 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = False 
                app.goodJob = False 
                app.visualize = True
                app.practice = False

    def displayM(app,canvas):
        if app.displayMessage == True and app.decode == False:
            canvas.create_text(app.width/2, 220, text = f'Plain Text: {app.message}', font = 'Times 30 bold', fill = 'dark blue')
            canvas.create_text(app.width/2, 240, text = f'Dictionary Key\n{app.key}', font = 'Times 20 bold', fill = 'dark green')
        if app.displayMessage == True and app.decode == True:
            canvas.create_text(app.width/2, 350, text = f'Plain Text: {app.message}', font = 'Times 30 bold', fill = 'dark blue')
            canvas.create_text(app.width/2, 400, text = f'Dictionary Key\n{app.key}', font = 'Times 20 bold', fill = 'dark green')
    
    def displayI(app,canvas):
        if app.displayIndex == True and app.decode == False: 
            key = app.key
            result = "" 
            for char in range(len(app.message)):
                letter = app.message[char]
                nL = letter.lower()
                if nL in key:
                    if letter.isupper() == False:
                        keyIndex = key[letter]
                        result += keyIndex
                    else:
                        lower = letter.lower()
                        kIndex = key[lower]
                        result += kIndex
                else:
                    result += letter
            canvas.create_text(app.width/2, 330, text = 'Using the dictionary key, find the corresponding value of the letter in the message.\nThen compile the letters to make the encrypted message', fill = 'black', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 465, text = f'Result: {result}', fill = 'red', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 600, text = "Press 'd' to visualize decoding Monoalphabetic Substitution Cipher", fill = 'purple', font = 'Times 30 bold')
        if app.displayIndex == True and app.decode == True: 
            Key = app.key
            invertedKey = {value : k for k, value in Key.items()}
            result = "" 

            # finds the key associated with the value from the inverted dictionary 
            for char in range(len(app.message)):
                letter = app.message[char]
                nL = letter.lower()
                if nL in Key:
                    if letter.isupper() == False:
                        keyIndex = invertedKey.get(letter, 0 )
                        result += keyIndex
                    else:
                        lower = letter.lower()
                        kIndex = invertedKey.get(lower, 0 )
                        result += kIndex
                else:
                    result += letter
            canvas.create_text(app.width/2, 480, text = 'Using the dictionary key, find the value of the letter in the encrypted message then find the corresponding key.\nThen compile the letters to make the encrypted message', fill = 'black', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 555, text = f'Result: {result}', fill = 'red', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 600, text = "Press 's' to practice decoding and encoding Monoalphabetic Substitution Cipher", fill = 'purple', font = 'Times 30 bold')
            

    def instructionPractice(app,canvas):
        if app.practice == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Polygram Substitution Practice', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 165, fill = 'black', text = 'Press on the green button to practice encoding, blue for decoding,\n\tand the purple to exit the practice', font = 'Times 30')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 - 100, app.width/2 - 150, app.height/2 + 100, fill = 'green')
            canvas.create_rectangle(app.width/2 + 150, app.height/2 - 100, app.width/2 + 350, app.height/2 + 100, fill = 'light blue')
            canvas.create_rectangle(app.width/2 - 100, app.height/2 + 150, app.width/2 + 100, app.height/2 + 300, fill = 'purple')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 + 150, app.width/2 - 200, app.height/2 + 300, fill = 'pink')
            canvas.create_rectangle(app.width/2 + 200, app.height/2 + 150, app.width/2 + 350, app.height/2 + 300, fill = 'orange')
            canvas.create_text(app.width/2 - 250, app.height/2, text = 'Practice\nEncoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 + 250, app.height/2, text = 'Practice\nDecoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2, app.height/2 + 225, text = 'Exit', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 - 275, app.height/2 + 225, text = 'Visualize\nDecoding', font = 'Times 25 bold', fill = 'black')
            canvas.create_text(app.width/2 + 275, app.height/2 + 225, text = 'Visualize\nEncoding', font = 'Times 25 bold', fill = 'black')

    
    def celebration(app,canvas):
        if app.goodJob == True:
            canvas.create_text(app.width/2, app.height/2, text = 'Congrats!', font = 'Times 30 bold', fill = 'pink')
    
    def redrawAll(app, canvas):
        app.title(canvas)
        app.displayM(canvas)
        app.displayI(canvas)
        app.instructionPractice(canvas)
        app.celebration(canvas)

def hillCipherWithKey(message):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    message = message.replace(' ', "")
    strippedMessage = ""
    for i in range(len(message)):
        lower = message[i].lower()
        if lower in alphabet:
            strippedMessage += lower
    rows = cols = len(message)
    keyMatrix = [([0] * cols) for row in range(rows)]
    for row in range(len(keyMatrix)):
        for col in range(len(keyMatrix[0])):
            keyMatrix[row][col] = random.randrange(0,26)
    messageVector = [([0] * 1) for row in range(len(message))]
    for row in range(len(messageVector)):
        for col in range(len(messageVector[0])):
            letter = message[row]
            index = alphabet.find(letter)
            messageVector[row][col] = index
    newVector = matrixMultiplication(keyMatrix, messageVector)
    modVector = []
    for val in range(len(newVector)):
        num = newVector[val] % 26
        modVector += [num]
    result = ""
    for num in range(len(modVector)):
        index = modVector[num]
        result += alphabet[index]
    return result, keyMatrix

class hillPractice(Mode):
    def appStarted(app):
        app.message = ''
        app.key = ''
        app.displayMessage = False 
        app.count = 0
        app.final = False 
        app.decode = False 
        app.visualize = True
        app.practice = False 
        app.goodJob = False 
        app.km = []
        app.v = []

    def title(app, canvas):
        if app.decode == True and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Hill Cipher Practice', font = 'Times 40 bold')
            instructionText = "\t        The Hill cipher is a polygram subsitituition cipher.\n\tTo decrypt, first find the modular inverse of the matrix key then\n\t                        multiply by the result vector.\n\tThis result will then be moded by 26 to get the decrypted message"
            canvas.create_text(app.width/2, 180, fill = 'black', text = f'{instructionText}', font = 'Times 25')
            canvas.create_text(app.width/2, 250, fill = 'black', text = 'Example', font = 'Times 30 bold underline')
        if app.decode == False and app.visualize == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Hill Cipher Practice', font = 'Times 40 bold')
            instructionText = "\tThe Hill cipher is a polygram subsitituition cipher.\n       The encryptor uses a key, that is transoformed into a matrix.\nThis matrix is then multiplied by the vector of the encrypted message.\nThe resultiing vector is moded by 26 then translated back into letters."
            canvas.create_text(app.width/2, 190, fill = 'black', text = f'{instructionText}', font = 'Times 25')
            canvas.create_text(app.width/2, 270, fill = 'black', text = 'Example', font = 'Times 30 bold underline')

    def keyPressed(app,event):
        if event.key == 'd':
            app.message = ''
            app.key = ''
            app.displayMessage = False 
            app.count = 0
            app.final = False 
            app.decode = True 
            app.visualize = True
            app.practice = False 
            app.goodJob = False 
            app.km = []
            app.v = [] 
        if event.key == 's':
            app.message = ''
            app.key = ''
            app.displayMessage = False 
            app.count = 0
            app.final = False 
            app.decode = False
            app.visualize = False
            app.goodJob = False 
            app.practice = True 
            app.km = []
            app.v = [] 

    def mousePressed(app, event):
        if event.x and app.count == 0  and app.practice == False:
            if app.decode == False:
                app.message = app.getUserInput('enter a word')
                app.key = app.getUserInput("Enter a key string that is the len of the word squared")
                while len(app.key) != len(app.message)**2:
                    app.key = app.getUserInput("Enter a key string that is the len of the word squared")
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                app.displayMessage = True 
                app.count += 1
            if app.decode == True:
                app.message = app.getUserInput('enter an ecrypted word')
                app.key = app.getUserInput("Enter the key string")
                while len(app.key) != len(app.message)**2:
                    app.key = app.getUserInput("Enter a key string that is the len of the encrypted word squared")
                if app.message == None:
                    app.app.setActiveMode(app.app.practiceMode)
                app.displayMessage = True 
                app.count += 1
        if event.x and app.count == 1 and app.practice == False:
            app.displayIndex = True 
            app.displayShift = True
            app.displaySD = True 
            app.final = True 
        if app.practice == True:
            if app.width/2 - 350 <= event.x <= app.width/2 - 150 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                userEncode = app.getUserInput('Enter a message to encode')
                app.key = app.getUserInput('Enter a key for your message(matrix)')
                userEncodedM = app.getUserInput('Enter your encrypted message')
                if userEncode == decodeHillCipher(userEncodedM, app.key):
                    app.goodJob = True 
                while userEncode != decodeHillCipher(userEncodedM, app.key) and userEncode != None:
                    userEncode = app.getUserInput(f'Try again! Enter encoded message for {message} with a key dictionary of {app.key}')
                    if userEncode == decodeHillCipher(userEncodedM, app.key):
                        app.goodJob = True 
                if userEncode == None:
                    app.app.setActiveMode(app.app.levelPage)
            if app.width/2 + 150 <= event.x <=  app.width/2 + 350 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                app.goodJob = False 
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                encryptedMessage, app.key = hillCipherWithKey(message)
                userDecode = app.getUserInput(f'Decrypt {encryptedMessage} with a key of {app.key}')
                if userDecode == message:
                    app.goodJob = True 
                while userDecode != message and userDecode != None:
                    userEncode = app.getUserInput(f'Try again! Decrypt {encryptedMessage} with a key of {app.key}')
                    if userDecode == message:
                        app.goodJob = True 
                if userDecode == None:
                    pass
            if app.width/2 - 100 <= event.x <= app.width/2 + 100 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.goodJob = False
                app.app.setActiveMode(app.app.practiceMode)
            if app.width/2 - 350 <= event.x <= app.width/2 - 200 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = True 
                app.goodJob = False 
                app.visualize = True
                app.practice = False
            if app.width/2 + 200 <= event.x <= app.width/2 + 350 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.message = ''
                app.shift = 0
                app.displayMessage = False 
                app.count = 0
                app.displayIndex = False
                app.lAndI = dict()
                app.displayShift = False 
                app.shiftDict = dict()
                app.displaySD = False
                app.final = False 
                app.decode = False 
                app.goodJob = False 
                app.visualize = True
                app.practice = False

    def displayM(app,canvas):
        if app.displayMessage == True and app.decode == False:
            row = len(app.message)
            col = len(app.message)
            # list coprehension https://www.cs.cmu.edu/~112/notes/notes-2d-lists.html
            app.km = [([0] * col) for r in range(row)]
            count = 0
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            for r in range(row):
                for c in range(col):
                    letter = app.key[count]
                    index = alphabet.find(letter)
                    app.km[r][c] = index
                    count += 1
            canvas.create_text(app.width/2, 305, text = f'Plain Text: {app.message}', font = 'Times 30 bold', fill = 'dark blue')
            canvas.create_text(app.width/2, 355, text = f'               Dictionary Key\n{app.km}', font = 'Times 30 bold', fill = 'dark green')
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            v = []
            for i in range(len(app.message)):
                letter = app.message[i]
                index = alphabet.find(letter)
                v.append([index])
            canvas.create_text(app.width/2, 425, text = 'Create a vector constructed by the\nindex of each letter of the plain text', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 475, text = f'Message vector: {v}', font = 'Times 30 bold', fill = 'dark green')
            resultKey = matrixMultiplication(app.km, v)
            canvas.create_text(app.width/2, 525, text = f'Multiply the key matrix and the message vector\n\t           {resultKey}', font = 'Times 30 bold', fill = 'black')
            resultV = []
            for r in range(len(resultKey)):
                    val = resultKey[r]
                    nV = val%26
                    resultV.append(nV)
            canvas.create_text(app.width/2, 575, text = 'Mod every value in the result vectory by 26', fill = 'black', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 605, text = f'{resultV}', fill = 'dark green', font = 'Times 30 bold')
            r = ''
            for i in range(len(resultV)):
                num = resultV[i]
                letter = alphabet[num]
                r += letter
            canvas.create_text(app.width/2, 640, text = 'Find the letter using the index from the vector', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 690, text = f'Result: {r}', fill = 'red', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 740, text = "Press 'd' to visualize decoding", fill = 'purple', font = 'Times 30 bold')
        if app.displayMessage == True and app.decode == True:
            row = len(app.message)
            col = len(app.message)
            # list coprehension https://www.cs.cmu.edu/~112/notes/notes-2d-lists.html
            app.km = [([0] * col) for r in range(row)]
            count = 0
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            for r in range(row):
                for c in range(col):
                    letter = app.key[count]
                    index = alphabet.find(letter)
                    app.km[r][c] = index
                    count += 1
            canvas.create_text(app.width/2, 290, text = f'Encrypted Text: {app.message}', font = 'Times 30 bold', fill = 'dark blue')
            canvas.create_text(app.width/2, 340, text = f'               Dictionary Key\n{app.km}', font = 'Times 30 bold', fill = 'dark green')
            inverseM = modMatInv(app.km,26)
            row = len(app.message)
            inverseMat = [([0] * row) for r in range(row)]
            for r in range(row):
                for c in range(row):
                    num = int(inverseM[r][c])
                    inverseMat[r][c] = num
            canvas.create_text(app.width/2, 400, text = 'Find the modular inverse matrix of the key matrix', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 435, text = f'{inverseMat}', font = 'Times 30 bold', fill = 'dark green')
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            v = []
            for i in range(len(app.message)):
                letter = app.message[i]
                index = alphabet.find(letter)
                v.append([index])
            canvas.create_text(app.width/2, 480, text = '              Create a vector constructed by the\nindex of each letter of the encrypted message', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 535, text = f'Encrypted message vector: {v}', font = 'Times 30 bold', fill = 'dark green')
            resultKey = matrixMultiplication(inverseMat, v)
            canvas.create_text(app.width/2, 585, text = f'Multiply the key matrix and the encrypted message vector\n\t           {resultKey}', font = 'Times 30 bold', fill = 'black')
            resultV = []
            for r in range(len(resultKey)):
                val = resultKey[r]
                nV = val%26
                resultV.append(nV)
            canvas.create_text(app.width/2, 635, text = 'Mod every value in the result vectory by 26', fill = 'black', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 665, text = f'{resultV}', fill = 'dark green', font = 'Times 30 bold')
            r = ''
            for i in range(len(resultV)):
                num = resultV[i]
                letter = alphabet[num]
                r += letter
            canvas.create_text(app.width/2, 700, text = 'Find the letter using the index from the vector', font = 'Times 30 bold', fill = 'black')
            canvas.create_text(app.width/2, 740, text = f'Result: {r}', fill = 'red', font = 'Times 30 bold')
            canvas.create_text(app.width/2, 780, text = "Press 's' to practice encoding and decoding Hill", fill = 'purple', font = 'Times 30 bold')
    
        
    def instructionPractice(app,canvas):
        if app.practice == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Hill Cipher Practice', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 165, fill = 'black', text = 'Press on the green button to practice encoding, blue for decoding,\n\tand the purple to exit the practice', font = 'Times 30')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 - 100, app.width/2 - 150, app.height/2 + 100, fill = 'green')
            canvas.create_rectangle(app.width/2 + 150, app.height/2 - 100, app.width/2 + 350, app.height/2 + 100, fill = 'light blue')
            canvas.create_rectangle(app.width/2 - 100, app.height/2 + 150, app.width/2 + 100, app.height/2 + 300, fill = 'purple')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 + 150, app.width/2 - 200, app.height/2 + 300, fill = 'pink')
            canvas.create_rectangle(app.width/2 + 200, app.height/2 + 150, app.width/2 + 350, app.height/2 + 300, fill = 'orange')
            canvas.create_text(app.width/2 - 250, app.height/2, text = 'Practice\nEncoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 + 250, app.height/2, text = 'Practice\nDecoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2, app.height/2 + 225, text = 'Exit', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 - 275, app.height/2 + 225, text = 'Visualize\nDecoding', font = 'Times 25 bold', fill = 'black')
            canvas.create_text(app.width/2 + 275, app.height/2 + 225, text = 'Visualize\nEncoding', font = 'Times 25 bold', fill = 'black')

    
    def celebration(app,canvas):
        if app.goodJob == True:
            canvas.create_text(app.width/2, app.height/2, text = 'Congrats!', font = 'Times 30 bold', fill = 'pink')
    
    def redrawAll(app, canvas):
        app.title(canvas)
        app.displayM(canvas)
        app.instructionPractice(canvas)
        app.celebration(canvas)
    

# getCellbounds is from the 15112 website - I just changed variables to fit my screen
# https://www.cs.cmu.edu/~112/notes/notes-animations-part1.html#exampleGrids

def getCellBounds(row, col):
    # aka 'modelToView'
    # returns (x0, y0, x1, y1) corners/bounding box of given cell in grid
    gridWidth  = 800 - 2*300
    gridHeight = 800 - 2*300
    x0 = 300 + gridWidth * col / 5
    x1 = 300 + gridWidth * (col+1) / 5
    y0 = 300 + gridHeight * row / 5
    y1 = 300 + gridHeight * (row+1) / 5
    return (x0, y0, x1, y1)

class playfairPractice(Mode):
    def appStarted(app):
        app.stepCount = 0 
        app.practice = False 
        app.modeType = 'Example'
        app.celebration = False 
        app.phrase = ''
        app.len = 0
        app.strippedPhrase = prepWord(app.phrase).replace(" ",'')
        app.positionDict1 = dict()
        app.positionDict2 = dict()
        app.prepedWord = ''

    def title(app, canvas):
        if app.practice == False:
            canvas.create_text(app.width/2, 30, fill = 'black', text = 'Playfair Cipher Practice', font = 'Times 40 bold')
            instructionText = "\t                To encode a Playfair cipher, first construct a 5x5 board\n     with letters A-Z, excluding Q. Then take the desired message and cut the words into pairs.\n       Locate the pairs and formulate boxes connecting the pairs. Then follow the rules\n             displayed bellow to find the encrypted letters to make the encrypted word.\n              Press 'f' to step through the example, press 'b' to step back and 'd' to decode "
            canvas.create_text(app.width/2, 110, fill = 'black', text = f'{instructionText}', font = 'Times 20')
            canvas.create_text(app.width/2, 180, fill = 'black', text = f'{app.modeType}', font = 'Times 35 bold underline')
    
    def keyPressed(app,event):
        if event.key == 'f':
            if app.stepCount == 0:
                app.phrase = app.getUserInput('Enter a word or two')
                stripped = app.phrase.replace(" ", "")
                app.len = len(stripped)
            if app.stepCount < 8:
                app.stepCount += 1
            if app.stepCount == 8:
                app.modeType = 'Practice'
                app.practice = True
                app.stepCount = 0
        if event.key == 'b':
            if app.stepCount > 0:
                app.stepCount -= 1
    
    def firstStep(app,canvas):
        canvas.create_text(app.width/2, 220, text= f'{app.phrase}', font = 'Times 30 bold', fill = 'red')
    
    def secondStep(app, canvas):
        if app.len%2 == 0:
            canvas.create_text(app.width/2, 245, text = 'split the word into pairs of letter', font = 'Times 25', fill = 'black')
        if app.len%2 == 1:
            canvas.create_text(app.width/2, 265, text = "split the word into pairs of letter and add an 'x' to the last letter without a pair and\n\treplace the second leter or a repeating pair with an 'x'", font = 'Times 23', fill = 'black')
    def thirdStep(app,canvas):
        app.prepedWord = prepWord(app.phrase)
        if app.len% 2 == 0:
            canvas.create_text(app.width/2, 275, text = f'{app.prepedWord}', font = 'Times 30 bold', fill = 'red')
        if app.len%2 != 0:
            canvas.create_text(app.width/2, 305, text = f'{app.prepedWord}', font = 'Times 30 bold', fill = 'red')
    def fourthStep(app,canvas):
        if app.len%2 == 0:
            canvas.create_text(app.width/2, 315, text = "Create a 5x5 grid with the alphabet, exlcluding 'q'", font = 'Times 25', fill = 'black')
        if app.len%2 != 0:
            canvas.create_text(app.width/2, 340, text = "Create a 5x5 grid with the alphabet, exlcluding 'q'", font = 'Times 25', fill = 'black')
    def fifthStep(app,canvas):
        alphabet = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'
        rows = 5
        cols = 5 
        index = 0
        if app.len%2 == 0:
            gridWidth  = 800 - 2*300
            gridHeight = 800 - 2*300
            cellWidth = gridWidth / 5
            cellHeight = gridHeight / 5
            for row in range(rows):
                for col in range(cols):
                    x0 = 310 + col * cellWidth
                    x1 = 310 + (col+1) * cellWidth
                    y0 = 335 + row * cellHeight
                    y1 = 335 + (row+1) * cellHeight
                    app.positionDict1[alphabet[index]] = (x0 + x1)/2, (y0 + y1)/2
                    canvas.create_rectangle(x0,y0,x1,y1, outline = 'black', width = 3)
                    canvas.create_text((x0 + x1)/2, (y0 + y1)/2, text = f'{alphabet[index]}', font = 'Times 15 bold', fill ='black' )
                    index += 1
        if app.len%2 != 0:
            gridWidth  = 800 - 2*300
            gridHeight = 800 - 2*300
            cellWidth = gridWidth / 5
            cellHeight = gridHeight / 5
            for row in range(rows):
                for col in range(cols):
                    x0 = 310 + col * cellWidth
                    x1 = 310 + (col+1) * cellWidth
                    y0 = 355 + row * cellHeight
                    y1 = 355 + (row+1) * cellHeight
                    app.positionDict2[alphabet[index]] = (x0 + x1)/2, (y0 + y1)/2
                    canvas.create_rectangle(x0,y0,x1,y1, outline = 'black', width = 3)
                    canvas.create_text((x0 + x1)/2, (y0 + y1)/2, text = f'{alphabet[index]}', font = 'Times 15 bold', fill ='black' )
                    index += 1

    def sixthStep(app,canvas):
        if app.len%2 == 0:
            canvas.create_text(app.width/2, 555, text = "Create boxes encompassing the letter pairs on the grid", font = 'Times 25 ', fill = 'black')
            alphabet = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'
            rows = 5
            cols = 5 
            index = 0
            letterPoint = []
            gridWidth  = 800 - 2*300
            gridHeight = 800 - 2*300
            cellWidth = gridWidth / 5
            cellHeight = gridHeight / 5
            for row in range(rows):
                for col in range(cols):
                    x0 = 310 + col * cellWidth
                    x1 = 310 + (col+1) * cellWidth
                    y0 = 335 + row * cellHeight
                    y1 = 335 + (row+1) * cellHeight
                    index += 1
                    letterPoint.append([row, col])
            index = 0
            alphabetL = 'abcdefghijklmnoprstuvwxyz'
            stripped = app.prepedWord.replace(' ', '')
            pairs = []
            count = 0
            for i in range(len(stripped)//2):
                if i == 0:
                    pair = stripped[i] + stripped[i+1]
                    pairs.append(pair)
                if i > 0:
                    val = 2*i
                    pair = stripped[val] + stripped[val + 1]
                    pairs.append(pair)
            for pair in range(len(pairs)):
                p = pairs[pair]
                fL, sL = p[0], p[1]
                indexFL = alphabetL.find(fL)
                indexSL = alphabetL.find(sL)
                r1, c1 = letterPoint[indexFL]
                r2 , c2 = letterPoint[indexSL]
                fx0, fy0, fx1, fy1 = getCellBounds(r1, c1)
                sx0, sy0, sx1, sy1= getCellBounds(r2, c2)
                xlist = [fx0, fx1, sx0, sx1]
                ylist = [fy0, fy1, sy0, sy1]
                color = ['pink', 'red', 'blue', 'cyan', 'green', 'orange', 'purple']
                canvas.create_rectangle(min(xlist)+10 ,min(ylist)+35, max(xlist)+10, max(ylist)+35, outline = color[index], width = 4 )
                index += 1
                
        if app.len%2 != 0:
            canvas.create_text(app.width/2, 570, text = "Create boxes encompassing the letter pairs on the grid", font = 'Times 25 ', fill = 'black')
            alphabet = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'
            rows = 5
            cols = 5 
            index = 0
            letterPoint = []
            gridWidth  = 800 - 2*300
            gridHeight = 800 - 2*300
            cellWidth = gridWidth / 5
            cellHeight = gridHeight / 5
            for row in range(rows):
                for col in range(cols):
                    x0 = 310 + col * cellWidth
                    x1 = 310 + (col+1) * cellWidth
                    y0 = 335 + row * cellHeight
                    y1 = 335 + (row+1) * cellHeight
                    index += 1
                    letterPoint.append([row, col])
            index = 0
            alphabetL = 'abcdefghijklmnoprstuvwxyz'
            stripped = app.prepedWord.replace(' ', '')
            pairs = []
            count = 0
            for i in range(len(stripped)//2):
                if i == 0:
                    pair = stripped[i] + stripped[i+1]
                    pairs.append(pair)
                if i > 0:
                    val = 2*i
                    pair = stripped[val] + stripped[val + 1]
                    pairs.append(pair)
            for pair in range(len(pairs)):
                p = pairs[pair]
                fL, sL = p[0], p[1]
                indexFL = alphabetL.find(fL)
                indexSL = alphabetL.find(sL)
                r1, c1 = letterPoint[indexFL]
                r2 , c2 = letterPoint[indexSL]
                fx0, fy0, fx1, fy1 = getCellBounds(r1, c1)
                sx0, sy0, sx1, sy1= getCellBounds(r2, c2)
                xlist = [fx0, fx1, sx0, sx1]
                ylist = [fy0, fy1, sy0, sy1]
                color = ['pink', 'red', 'blue', 'cyan', 'green', 'orange', 'purple']
                canvas.create_rectangle(min(xlist)+10 ,min(ylist)+55, max(xlist)+10, max(ylist)+55, outline = color[index], width = 4 )
                index += 1
             
    def seventhStep(app,canvas):
        if app.len%2 == 0:
            canvas.create_text(app.width/2, 600, text = '\tFind the letter in the opposing corner of each letter pair.\nIf box is vertical or horizontal, find the letter below(vertical) or to the right(horizontal)', font = 'Times 20 bold', fill ='black')
            result = PlayFairCipher(app.phrase)
            canvas.create_text(app.width/2, 660, text = f'Result: {result}', font = 'Times 30 bold', fill = 'red')
            canvas.create_text(app.width/2, 690, text = "Press 'f' to continue to practice encoding and decoding", fill = 'purple', font = 'Times 30 bold')
        if app.len%2 != 0:
            canvas.create_text(app.width/2, 625, text = '\tFind the letter in the opposing corner of each letter pair.\nIf box is vertical or horizontal, find the letter below(vertical) or to the right(horizontal)', font = 'Times 20 bold', fill ='black')
            result = PlayFairCipher(app.phrase)
            canvas.create_text(app.width/2, 675, text = f'Result: {result}', font = 'Times 30 bold', fill = 'red')
            canvas.create_text(app.width/2, 720, text = "Press 'f' to continue to practice encoding and decoding", fill = 'purple', font = 'Times 30 bold')
    
    def mousePressed(app,event):
        if app.practice == True:
            if app.width/2 - 350 <= event.x <= app.width/2 - 150 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                app.goodJob = False 
                userEncode = app.getUserInput('Enter a message to encode')
                userEncodedM = app.getUserInput('Enter your encrypted message')
                if userEncode == None or userEncodedM == None:
                    pass
                if userEncode == playFairCipherDecorder(userEncodedM):
                    app.goodJob = True 
                while userEncode != playFairCipherDecorder(userEncodedM) and userEncode != None and userEncodedM != None:
                    userEncode = app.getUserInput(f'Try again! Enter encoded message for {message} with a key dictionary of {app.key}')
                    if userEncode == decodeHillCipher(userEncodedM, app.key):
                        app.goodJob = True 
                if userEncode == None or userEncodedM == None:
                    app.app.setActiveMode(app.app.levelPage)
            if app.width/2 + 150 <= event.x <=  app.width/2 + 350 and app.height/2 - 100 <= event.y <= app.height/2 + 100:
                app.goodJob = False 
                randNum = random.randrange(0, 33)
                message = sentences[randNum]
                encryptedMessage = PlayFairCipher(message)
                userDecode = app.getUserInput(f'Decrypt {encryptedMessage}')
                if userDecode == message:
                    app.goodJob = True 
                while userDecode != message and userDecode != None:
                    userEncode = app.getUserInput(f'Try again! Decrypt {encryptedMessage} with a key of {app.key}')
                    if userDecode == message:
                        app.goodJob = True 
                if userDecode == None:
                    pass
            if app.width/2 - 100 <= event.x <= app.width/2 + 100 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.goodJob = False
                app.app.setActiveMode(app.app.practiceMode)
            if app.width/2 - 350 <= event.x <= app.width/2 - 200 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.practice = False
                app.stepCount = 0
            if app.width/2 + 200 <= event.x <= app.width/2 + 350 and app.height/2 + 150 <= event.y <= app.height/2 + 300:
                app.practice = False
                app.stepCount = 0

    def instructionPractice(app,canvas):
        if app.practice == True:
            canvas.create_text(app.width/2, 100, fill = 'black', text = 'Playfair Practice', font = 'Times 50 bold')
            canvas.create_text(app.width/2, 165, fill = 'black', text = 'Press on the green button to practice encoding, blue for decoding,\n\tand the purple to exit the practice', font = 'Times 30')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 - 100, app.width/2 - 150, app.height/2 + 100, fill = 'green')
            canvas.create_rectangle(app.width/2 + 150, app.height/2 - 100, app.width/2 + 350, app.height/2 + 100, fill = 'light blue')
            canvas.create_rectangle(app.width/2 - 100, app.height/2 + 150, app.width/2 + 100, app.height/2 + 300, fill = 'purple')
            canvas.create_rectangle(app.width/2 - 350, app.height/2 + 150, app.width/2 - 200, app.height/2 + 300, fill = 'pink')
            canvas.create_rectangle(app.width/2 + 200, app.height/2 + 150, app.width/2 + 350, app.height/2 + 300, fill = 'orange')
            canvas.create_text(app.width/2 - 250, app.height/2, text = 'Practice\nEncoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 + 250, app.height/2, text = 'Practice\nDecoding', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2, app.height/2 + 225, text = 'Exit', fill = 'white', font = 'Times 25 bold')
            canvas.create_text(app.width/2 - 275, app.height/2 + 225, text = 'Visualize\nDecoding', font = 'Times 25 bold', fill = 'black')
            canvas.create_text(app.width/2 + 275, app.height/2 + 225, text = 'Visualize\nEncoding', font = 'Times 25 bold', fill = 'black')

    
    def celebration(app,canvas):
        if app.goodJob == True:
            canvas.create_text(app.width/2, app.height/2, text = 'Congrats!', font = 'Times 30 bold', fill = 'pink')



    def redrawAll(app, canvas):
        app.title(canvas) 
        if app.stepCount == 1: 
            app.firstStep(canvas)
        if app.stepCount == 2:
            app.firstStep(canvas)
            app.secondStep(canvas)
        if app.stepCount == 3:
            app.firstStep(canvas)
            app.secondStep(canvas)
            app.thirdStep(canvas)
        if app.stepCount == 4:
            app.firstStep(canvas)
            app.secondStep(canvas)
            app.thirdStep(canvas)
            app.fourthStep(canvas)
        if app.stepCount == 5:
            app.firstStep(canvas)
            app.secondStep(canvas)
            app.thirdStep(canvas)
            app.fourthStep(canvas)
            app.fifthStep(canvas)
        if app.stepCount == 6:
            app.firstStep(canvas)
            app.secondStep(canvas)
            app.thirdStep(canvas)
            app.fourthStep(canvas)
            app.fifthStep(canvas)
            app.sixthStep(canvas)
        if app.stepCount == 7:
            app.firstStep(canvas)
            app.secondStep(canvas)
            app.thirdStep(canvas)
            app.fourthStep(canvas)
            app.fifthStep(canvas)
            app.sixthStep(canvas)
            app.seventhStep(canvas)
        if app.practice == True:
            app.instructionPractice(canvas)
        if app.celebration == True:
            app.celebration(canvas)


class MyGame(ModalApp):
    def appStarted(app):
        app.myIntroPage = myIntroPage()
        app.instructionPage = instructionPage()
        app.levelPage = levelPage()
        app.setActiveMode(app.instructionPage)
        app.practiceMode = practiceMode()
        app.gameOver = gameOver()
        app.caesarPractice = caesarPractice()
        app.polygramPractice = polygramPractice()
        app.monoalphabeticPractice = monoalphabeticPractice()
        app.hillPractice = hillPractice()
        app.playfairPractice = playfairPractice()
        app.celebrationPage = celebrationPage()
        app.level = 1
        app.lives = 3 
        app.level1 = level1()
        app.level2 = level2()
        app.level3 = level3()
        app.level4 = level4()
        app.level5 = level5()
        app.level6 = level6()
        app.level7 = level7()
        app.level8 = level8()
        app.level9 = level9()
        app.level10 = level10()
        app.level11 = level11()
        app.level12 = level12()
        app.level13 = level13()
        app.level14 = level14()
        app.level15 = level15()
        app.level16 = level16()
        app.level17 = level17()
        app.level18 = level18()
        app.level19 = level19()
        app.level20 = level20()
        app.level21 = level21()
        app.level22 = level22()
        app.level23 = level23()
        app.level24 = level24()
        app.level25 = level25()
        app.level26 = level26()
        app.level27 = level27()
        app.level28 = level28()
        app.level29 = level29()
        app.level30 = level30()
        app.level31 = level31()
        app.level32 = level32()
        app.level33 = level33()
        app.level34 = level34()
        app.level35 = level35()
        app.level36 = level36()
        app.level37 = level37()
        app.level38 = level38()
        app.level39 = level39()
        app.level40 = level40()
        app.level41 = level41()
        app.level42 = level42()
        app.level43 = level43()
        app.level44 = level44()
        app.level45 = level45()
        app.level46 = level46()
        app.level47 = level47()
        app.level48 = level48()
        app.level49 = level49()
        

app = MyGame(width = 800, height = 800)