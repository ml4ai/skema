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
        # Create basic model table
        new_model_table_container_basic = self.soup.new_tag("div", id=f"{model_name}-basic" , class_="table-container")
        new_model_table_basic = self.soup.new_tag("table")
        new_model_table_header_basic = self.soup.new_tag("tr")
        new_model_table_header_basic_th1 = self.soup.new_tag('th')
        new_model_table_header_basic_th1.string = "File Name"
        new_model_table_header_basic_th2 = self.soup.new_tag('th')
        new_model_table_header_basic_th2.string = 'Num Lines'
        new_model_table_header_basic_th3 = self.soup.new_tag('th')
        new_model_table_header_basic_th3.string = 'Can Ingest'
        new_model_table_header_basic_th4 = self.soup.new_tag('th')
        new_model_table_header_basic_th4.string = 'Tree-Sitter Parse Tree'
        new_model_table_header_basic_th5 = self.soup.new_tag('th')
        new_model_table_header_basic_th5.string = 'CAST'
        new_model_table_header_basic_th6 = self.soup.new_tag('th')
        new_model_table_header_basic_th6.string = 'Gromet'

        # Append the elements to each other
        new_model_container.extend([new_model_heading, new_model_table_container_basic])
        new_model_table_container_basic.append(new_model_table_basic)
        new_model_table_basic.append(new_model_table_header_basic)
        new_model_table_header_basic.extend([new_model_table_header_basic_th1, new_model_table_header_basic_th2, new_model_table_header_basic_th3, new_model_table_header_basic_th4, new_model_table_header_basic_th5, new_model_table_header_basic_th6])
    
        # Append to outer body
        self.soup.body.append(new_model_container)

    def add_model_header_data(self, model: str, supported_lines: int, total_lines: int, full_ingest: bool):
        model_table = self.soup.select_one(f"#{model}-basic table") 

        model_header = self.soup.new_tag("div", class_="model_header") 
        model_header_supported_lines = self.soup.new_tag("div", class_="model_header") 
        model_header_supported_lines.string = f"Total Supported Lines: {supported_lines}/{total_lines}"
        model_header_full_ingest = self.soup.new_tag("div") 
        model_header_full_ingest.string = f"Can ingest system into single GrometFNModuleCollection: {str(full_ingest)}"
        

        model_header.extend([model_header_supported_lines, model_header_full_ingest])
        model_table.append(model_header)
        
    def add_file_basic(self, model: str, file_name: str, num_lines: int, can_ingest: bool, parse_tree_path: Path, cast_path: Path, gromet_path: Path):
        model_table = self.soup.select_one(f"#{model}-basic table") 
        
        new_row = self.soup.new_tag("tr")
        
        file_name_cell = self.soup.new_tag("td")
        file_name_cell.string = file_name

        num_lines_cell = self.soup.new_tag("td")
        num_lines_cell.string = str(num_lines)

        valid_cell = self.soup.new_tag("td")
        valid_cell.string = "✓" if can_ingest else "✗"

        parse_tree_cell = self.soup.new_tag("td")
        parse_tree_a = self.soup.new_tag("a")
        parse_tree_a["href"] = str(parse_tree_path)
        parse_tree_a.string = "Open Parse Tree"
        parse_tree_cell.append(parse_tree_a)

        cast_cell = self.soup.new_tag("td")
        cast_a = self.soup.new_tag("a")
        cast_a["href"] = str(cast_path)
        cast_a.string = "Open CAST"
        cast_cell.append(cast_a)

        gromet_cell = self.soup.new_tag("td")
        gromet_a = self.soup.new_tag("a")
        gromet_a["href"] = str(gromet_path)
        gromet_a.string = "Open Gromet"
        gromet_cell.append(gromet_a)

        new_row.extend([file_name_cell, num_lines_cell, valid_cell, parse_tree_cell, cast_cell, gromet_cell])

        model_table.append(new_row)

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