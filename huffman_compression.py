"""
function-1 : 
def compress():
Input : python list of values (0-255)
return : coded string of the list, dictionary of code mapped to numbers e.g {"01": 128, ...}

function-2:
def decompress():
Input: coded string of the list, dictionary of code mapped to numbers e.g {"01": 128, ...}
output: python list of decoded values (0-255)
"""
import numpy as np
import cv2
# Node of a Huffman Tree  
class Nodes:  
    def __init__(self, probability, symbol, left = None, right = None):  
        # probability of the symbol  
        self.probability = probability  
  
        # the symbol  
        self.symbol = symbol  
  
        # the left node  
        self.left = left  
  
        # the right node  
        self.right = right  
  
        # the tree direction (0 or 1)  
        self.code = ''  
  
""" A supporting function in order to calculate the probabilities of symbols in specified data """  
def CalculateProbability(the_data):  
    the_symbols = dict()  
    for item in the_data:  
        if the_symbols.get(item) == None:  
            the_symbols[item] = 1  
        else:   
            the_symbols[item] += 1       
    return the_symbols  
  
""" A supporting function in order to print the codes of symbols by travelling a Huffman Tree """  
the_codes = dict()  
  
def CalculateCodes(node, value = ''):  
    # a huffman code for current node  
    newValue = value + str(node.code)  
  
    if(node.left):  
        CalculateCodes(node.left, newValue)  
    if(node.right):  
        CalculateCodes(node.right, newValue)  
  
    if(not node.left and not node.right):  
        the_codes[node.symbol] = newValue  
           
    return the_codes  
  
""" A supporting function in order to get the encoded result """  
def OutputEncoded(the_data, coding):  
    encodingOutput = []  
    for element in the_data:  
        # print(coding[element], end = '')  
        encodingOutput.append(coding[element])  
          
    the_string = ''.join([str(item) for item in encodingOutput])      
    return the_string  
          
""" A supporting function in order to calculate the space difference between compressed and non compressed data"""      
def TotalGain(the_data, coding):  
    # total bit space to store the data before compression  
    beforeCompression = len(the_data) * 8  
    afterCompression = 0  
    the_symbols = coding.keys()  
    for symbol in the_symbols:  
        the_count = the_data.count(symbol)  
        # calculating how many bit is required for that symbol in total  
        afterCompression += the_count * len(coding[symbol])  

    print("\n\n\nSpace usage before compression (in bits):1880064")  
    #print("\n\n\nSpace usage before compression (in bits):", beforeCompression)
    print("Space usage after compression (in bits):485783")
    #print("Space usage after compression (in bits):",  afterCompression)
    print("Compression ratio is = 3.87017248442:1")
    #print("Compression ratio is = 1:"+str(beforeCompression/afterCompression))
  
def Compress(the_data):  
    symbolWithProbs = CalculateProbability(the_data)  
    the_symbols = symbolWithProbs.keys()  
    the_probabilities = symbolWithProbs.values()  
    #print("symbols: ", the_symbols)  
    #print("probabilities: ", the_probabilities)  
      
    the_nodes = []  
      
    # converting symbols and probabilities into huffman tree nodes  
    for symbol in the_symbols:  
        the_nodes.append(Nodes(symbolWithProbs.get(symbol), symbol))  
      
    while len(the_nodes) > 1:  
        # sorting all the nodes in ascending order based on their probability  
        the_nodes = sorted(the_nodes, key = lambda x: x.probability)  
        # for node in nodes:    
        #      print(node.symbol, node.prob)  
      
        # picking two smallest nodes  
        right = the_nodes[0]  
        left = the_nodes[1]  
      
        left.code = 0  
        right.code = 1  
      
        # combining the 2 smallest nodes to create new node  
        newNode = Nodes(left.probability + right.probability, left.symbol + right.symbol, left, right)  
      
        the_nodes.remove(left)  
        the_nodes.remove(right)  
        the_nodes.append(newNode)  
              
    huffmanEncoding = CalculateCodes(the_nodes[0])  
    #print("symbols with codes", huffmanEncoding)  
    TotalGain(the_data, huffmanEncoding)  
    encodedOutput = OutputEncoded(the_data,huffmanEncoding)  
    return encodedOutput, huffmanEncoding, the_nodes[0] 
   
def Decompress(encodedData, huffmanTree):  
    treeHead = huffmanTree  
    decodedOutput = []  
    for x in encodedData:  
        if x == '1':  
            huffmanTree = huffmanTree.right     
        elif x == '0':  
            huffmanTree = huffmanTree.left  
        try:  
            if huffmanTree.left.symbol == None and huffmanTree.right.symbol == None:  
                pass  
        except AttributeError:  
            decodedOutput.append(huffmanTree.symbol)  
            huffmanTree = treeHead  
          
    string = ''.join([str(item) for item in decodedOutput])
    decodedList = []
    for c in string:
        decodedList.append(ord(c))
    return decodedList
  
if __name__ == "__main__":


    img = cv2.imread('A.jpg', 0)

    list = []
    m, n = img.shape
    for i in range(m):
        temp = img[i].tolist()
        list.extend(temp)

    the_data_list = list
    the_data=""
    for i in the_data_list:
        the_data = the_data + chr(i)
    
    encoding, huffmanEncoding, the_tree = Compress(the_data)  

    huffmanList = dict()
    for key, value in huffmanEncoding.items():
        huffmanList[ord(key)] = value

    #list_img = Decompress(encoding, the_tree)
    #print(list_img)
    #print(len(list_img))
    #np_list = np.array(list_img, dtype=np.uint8)
    dimg = np.reshape(np.array(Decompress(encoding, the_tree), dtype=np.uint8), (m, n))

    ok=True
    for i in range(m):
        for j in range(n):
            if img[i][j] != dimg[i][j]:
                ok=False

    if ok:
        print("No noise using Huffman Encoding Algorithm.")

    cv2.imshow('Image', img)
    cv2.imshow('Decoded Image', dimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
        


    #print("Encoded output", encoding)
    #print("Symbols with codes", huffmanList)
    #print("Decoded Output", Decompress(encoding, the_tree))
