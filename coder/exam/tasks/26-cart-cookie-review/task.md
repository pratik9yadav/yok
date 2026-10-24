We're about to launch the online order page for our shop, and `cart_session.py` is the part that keeps the customer's cart in a browser cookie and reads the quantity they type into the box.
A friend wrote it and it works in our testing, but nobody has reviewed it.
Please review it the way you would before something goes live on the public internet, and fix whatever you wouldn't let ship.
Keep the functions and how they are called (`save_cart`, `load_cart`, `parse_qty`, `add_item`, `MAX_QTY`); normal customers shouldn't notice a difference, and typing `2*12` for two dozen must keep working.
It's fine if carts saved by the old version stop loading after the launch.
