-- Initialize database with sample data for the n8n agent

-- Create a sample table for products
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2),
    category VARCHAR(100),
    stock_quantity INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create a sample table for customers
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    city VARCHAR(100),
    country VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create a sample table for orders
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    status VARCHAR(50) DEFAULT 'pending',
    shipping_address TEXT
);

-- Insert sample data
INSERT INTO products (name, description, price, category, stock_quantity) VALUES
    ('Laptop Pro 15', 'High-performance laptop with 16GB RAM and 512GB SSD', 1299.99, 'Electronics', 25),
    ('Wireless Mouse', 'Ergonomic wireless mouse with precision tracking', 29.99, 'Accessories', 150),
    ('USB-C Hub', 'Multi-port USB-C hub with HDMI, USB 3.0, and SD card reader', 49.99, 'Accessories', 80),
    ('Mechanical Keyboard', 'RGB mechanical keyboard with blue switches', 89.99, 'Accessories', 45),
    ('4K Webcam', 'Professional 4K webcam with auto-focus and noise cancellation', 149.99, 'Electronics', 30),
    ('Monitor Stand', 'Adjustable monitor stand with cable management', 39.99, 'Furniture', 60),
    ('Desk Lamp', 'LED desk lamp with adjustable brightness and color temperature', 34.99, 'Furniture', 90),
    ('External SSD 1TB', 'Portable SSD with USB-C connection and 1TB capacity', 119.99, 'Storage', 55);

INSERT INTO customers (first_name, last_name, email, phone, city, country) VALUES
    ('Jan', 'Novák', 'jan.novak@email.cz', '+420123456789', 'Praha', 'Czech Republic'),
    ('Marie', 'Svobodová', 'marie.svobodova@email.cz', '+420987654321', 'Brno', 'Czech Republic'),
    ('Petr', 'Dvořák', 'petr.dvorak@email.cz', '+420111222333', 'Ostrava', 'Czech Republic'),
    ('Eva', 'Černá', 'eva.cerna@email.cz', '+420444555666', 'Plzeň', 'Czech Republic'),
    ('Tomáš', 'Procházka', 'tomas.prochazka@email.cz', '+420777888999', 'České Budějovice', 'Czech Republic');

INSERT INTO orders (customer_id, total_amount, status, shipping_address) VALUES
    (1, 1349.98, 'delivered', 'Václavské náměstí 1, 110 00 Praha'),
    (2, 169.98, 'shipped', 'Náměstí Svobody 10, 602 00 Brno'),
    (3, 89.99, 'processing', 'Masarykovo náměstí 5, 702 00 Ostrava'),
    (4, 199.98, 'pending', 'Náměstí Republiky 3, 301 00 Plzeň'),
    (1, 49.99, 'delivered', 'Václavské náměstí 1, 110 00 Praha');
