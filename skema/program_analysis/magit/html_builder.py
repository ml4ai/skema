from pathlib import Path 

from bs4 import BeautifulSoup

HTML_BASE_PATH = Path(__file__).parent / "base.html"
HTML_BASE = HTML_BASE_PATH.read_text()

class HTML_Instance():
    def __init__(self):
        self.soup = BeautifulSoup(HTML_BASE)
    
    def add_model(self, model_name: str):
        """Adds a new model"""
        # Create the new model HTML structure
        new_model_container = self.soup.new_tag("div", id=model_name, class_="model-container") 
        new_model_heading = self.soup.new_tag("h2")
        new_model_heading.string = model_name
        new_model_table_container = self.soup.new_tag("div", id=f"{model_name}-basic" , class_="table-container")
        new_model_table = self.soup.new_tag("table")
        new_model_table_header = self.soup.new_tag("tr")
        new_model_table_header_th1 = self.soup.new_tag('th')
        new_model_table_header_th1.string = "File Name"
        new_model_table_header_th2 = self.soup.new_tag('th')
        new_model_table_header_th2.string = 'Num Lines'
        new_model_table_header_th3 = self.soup.new_tag('th')
        new_model_table_header_th3.string = 'Can Ingest'

        # Append the elements to each other
        new_model_container.extend([new_model_heading, new_model_table_container])
        new_model_table_container.append(new_model_table)
        new_model_table.append(new_model_table_header)
        new_model_table_header.extend([new_model_table_header_th1, new_model_table_header_th2, new_model_table_header_th3])
        
        # Append to outer body
        self.soup.body.append(new_model_container)

    def add_file_basic(self, model: str, file_name: str, num_lines: int, can_ingest: bool):
        model_table = self.soup.select_one(f"#{model}-basic table") 
        
        new_row = self.soup.new_tag("tr")
        
        file_name_cell = self.soup.new_tag("td")
        file_name_cell.string = file_name

        num_lines_cell = self.soup.new_tag("td")
        num_lines_cell.string = str(num_lines)

        valid_cell = self.soup.new_tag("td")
        valid_cell.string = "✓" if can_ingest else "✗"

        new_row.extend([file_name_cell, num_lines_cell, valid_cell])

        model_table.append(new_row)

    def add_file_detailed():
        pass

    def add_file_experimental():
        pass

    

    def write_html(self):
        output_path = Path(__file__).resolve().parent / "output.html"
        output_path.write_text(self.soup.prettify())

"""
<div id="model1" class="model-container">
        <h2>Model 1</h2>
        <div class="table-container" id="model1-basic">
            <table>
                <tr>
                    <th>File Name</th>
                    <th>Basic Coverage</th>
                </tr>
                <!-- Add table rows for Model 1 - Basic -->
            </table>
        </div>
        <div class="table-container" id="model1-intermediate">
            <table>
                <tr>
                    <th>File Name</th>
                    <th>Intermediate Coverage</th>
                </tr>
                <!-- Add table rows for Model 1 - Intermediate -->
            </table>
        </div>
        <div class="table-container" id="model1-advanced">
            <table>
                <tr>
                    <th>File Name</th>
                    <th>Advanced Coverage</th>
                </tr>
                <!-- Add table rows for Model 1 - Advanced -->
            </table>
        </div>
    </div>
"""