#!/usr/bin/env python3

# This is a bad code example that violates clean code principles

class data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def process(self, z):
        # This function is too long and does too many things
        tmp = self.x + self.y
        if z > 0:
            tmp = tmp * z
        else:
            tmp = tmp / (z * -1)
        
        # Direct database access without error handling
        db = get_db_connection()
        db.execute(f"INSERT INTO results VALUES ({self.x}, {self.y}, {z}, {tmp})")
        
        # Complex nested conditionals
        if tmp > 100:
            if z > 10:
                if self.x > self.y:
                    print("Case 1")
                    return tmp + 10
                else:
                    print("Case 2")
                    return tmp + 20
            else:
                if self.x > self.y:
                    print("Case 3")
                    return tmp + 30
                else:
                    print("Case 4")
                    return tmp + 40
        else:
            if z > 10:
                if self.x > self.y:
                    print("Case 5")
                    return tmp + 50
                else:
                    print("Case 6")
                    return tmp + 60
            else:
                if self.x > self.y:
                    print("Case 7")
                    return tmp + 70
                else:
                    print("Case 8")
                    return tmp + 80
    
    def do_stuff(self):
        # Generic exception handling
        try:
            # Some complex operation
            result = self.x / self.y
            db = get_db_connection()
            db.execute(f"UPDATE data SET result = {result} WHERE id = {self.x}")
            return result
        except Exception:
            # Catching all exceptions without specific handling
            print("An error occurred")
            return None
