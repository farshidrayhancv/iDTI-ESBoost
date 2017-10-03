
class node:
    type = ''
    elements = []
    child_nodes = []


    def insert(self,x):
        if len( self.elements ) < 4:
            self.elements.append(x)
            print("inserted")
        else:
            print("error")
            length = len( self.elements )

            first_half = self.elements[0:int(length/2)]
            # print(first_half)
            second_half = self.elements[int(length/2):]

            node_1 = node()
            node_2 = node()

            node_1.elements = first_half
            node_2.elements = second_half

            node_1.type = 'parent'
            node_2.type = 'parent'


            self.type = 'root'
            self.child_nodes.append(node_1)
            self.child_nodes.append(node_2)
            self.elements = [node_2.elements[0]]

        self.elements.sort()
        return self

bucket = node()
bucket.type = "root"
# node_1 = node()
# node_2 = node()

bucket = bucket.insert(51)
bucket = bucket.insert(12)
bucket = bucket.insert(23)
bucket = bucket.insert(4)
bucket = bucket.insert(5)



