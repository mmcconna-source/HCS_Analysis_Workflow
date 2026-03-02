import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np
import string
from pathlib import Path
from typing import List, Dict, Optional, Union
import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os

class InteractivePlatePlanner:
    """
    Interactive widget for planning plate experiments (V2.1: Compound-Centric + Metadata).
    """
    
    def __init__(self, plate_type: int = 96):
        self.plate_type = plate_type
        self.rows = 8 if plate_type == 96 else 16
        self.cols = 12 if plate_type == 96 else 24
        self.row_labels = [string.ascii_uppercase[i] for i in range(self.rows)]
        self.col_labels = [str(i+1) for i in range(self.cols)]
        
        # Configuration
        self.compound_key = "Metadata_Drug"
        self.dose_key = "Metadata_Dose"
        
        # Data storage
        self.well_data = {} # {well_id: {key: value}}
        self.selected_wells = set()
        self.well_widgets = {}
        
        # Compound Library
        # Structure: {name: {'color': hex_code, 'meta': {key: val}}}
        self.compounds = {} 
        self.active_compound = None
        self.color_cycle = plt.get_cmap('tab20').colors # List of (r,g,b)
        self.color_idx = 0
        
        # Initialize UI Components
        self._create_ui()
        
    def _create_ui(self):
        """Builds the widget interface."""
        
        # --- 1. Plate Grid ---
        grid_items = []
        # Header
        grid_items.append(widgets.Label("")) 
        for c in self.col_labels:
            grid_items.append(widgets.Label(c, layout=widgets.Layout(width='30px', justify_content='center')))
            
        # Rows
        for r in self.row_labels:
            grid_items.append(widgets.Label(r, layout=widgets.Layout(width='20px')))
            for c in self.col_labels:
                well_id = f"{r}{int(c):02d}"
                btn = widgets.Button(
                    description='',
                    layout=widgets.Layout(width='30px', height='30px'),
                    style=widgets.ButtonStyle(button_color='lightgray'),
                    tooltip=f"Well: {well_id}"
                )
                btn.on_click(self._on_well_click)
                btn.well_id = well_id
                self.well_widgets[well_id] = btn
                grid_items.append(btn)
        
        self.plate_grid = widgets.GridBox(
            children=grid_items,
            layout=widgets.Layout(
                grid_template_columns=f"30px {' '.join(['30px']*self.cols)}",
                grid_gap='2px'
            )
        )
        
        # --- 2. Compound Manager ---
        self.txt_compound_name = widgets.Text(placeholder="New Compound Name")
        self.txt_compound_meta = widgets.Text(placeholder="Meta (e.g. Type=Ctrl; Batch=1)", description="Meta:")
        self.btn_add_compound = widgets.Button(description="Add", layout=widgets.Layout(width='60px'))
        self.btn_add_compound.on_click(self._add_compound)
        
        # CSV Import
        self.txt_import_path = widgets.Text(placeholder="compound_library_template.csv", description="CSV:")
        self.btn_import_csv = widgets.Button(description="Import", button_style='info', layout=widgets.Layout(width='70px'))
        self.btn_import_csv.on_click(self._import_csv_lib)

        self.dropdown_compounds = widgets.Dropdown(description="Active:", options=[])
        self.dropdown_compounds.observe(self._on_compound_select, names='value')
        
        self.compound_legend = widgets.Output(layout={'height': '80px', 'overflow_y': 'scroll', 'border': '1px solid #eee'})
        
        compound_box = widgets.VBox([
            widgets.HTML("<b>1. Compound Library</b>"),
            widgets.HBox([self.txt_compound_name, self.btn_add_compound]),
            self.txt_compound_meta,
            widgets.HBox([self.txt_import_path, self.btn_import_csv]),
            self.compound_legend,
            self.dropdown_compounds
        ], layout=widgets.Layout(border='1px solid #ccc', padding='5px', margin='5px 0'))
        
        # --- 3. Assignment Tools (Tabs) ---
        
        # Tab 1: Single/Manual Assignment
        self.input_manual_dose = widgets.FloatText(description="Dose:", step=0.1)
        self.btn_assign_manual = widgets.Button(description="Assign to Selected", button_style='primary')
        self.btn_assign_manual.on_click(self._assign_manual)
        
        tab_manual = widgets.VBox([
            widgets.HTML("<i>Assign active compound to selected wells.</i>"),
            self.input_manual_dose,
            self.btn_assign_manual
        ], layout=widgets.Layout(padding='10px'))
        
        # Tab 2: Dilution Series
        self.dil_start = widgets.FloatText(description="Start:", value=100)
        self.dil_factor = widgets.FloatText(description="Factor:", value=2)
        self.dil_steps = widgets.IntText(description="Steps:", value=8)
        self.dil_reps = widgets.IntText(description="Reps:", value=1)
        self.dil_dir = widgets.Dropdown(description="Dir:", options=['Horizontal', 'Vertical'], value='Horizontal')
        self.btn_gen_dilution = widgets.Button(description="Generate Series", button_style='info')
        self.btn_gen_dilution.on_click(self._generate_dilution)
        
        tab_dilution = widgets.VBox([
            widgets.HTML("<i>Select <b>ONE</b> start well. Series will fill automatically.</i>"),
            self.dil_start,
            self.dil_factor,
            self.dil_steps,
            self.dil_reps,
            self.dil_dir,
            self.btn_gen_dilution
        ], layout=widgets.Layout(padding='10px'))
        
        self.tabs = widgets.Tab(children=[tab_manual, tab_dilution])
        self.tabs.set_title(0, "Manual Assign")
        self.tabs.set_title(1, "Dilution Series")
        
        assignment_box = widgets.VBox([
            widgets.HTML("<b>2. Assignment</b>"),
            self.tabs
        ], layout=widgets.Layout(border='1px solid #ccc', padding='5px', margin='5px 0'))
        
        # --- 4. Plate Actions ---
        self.btn_select_all = widgets.Button(description="Select All")
        self.btn_select_all.on_click(self._select_all)
        self.btn_clear_sel = widgets.Button(description="Clear Selection")
        self.btn_clear_sel.on_click(self._clear_selection)
        self.btn_unassign = widgets.Button(description="Unassign Data", button_style='danger')
        self.btn_unassign.on_click(self._unassign_data)
        self.btn_randomize = widgets.Button(description="Randomize Plate", button_style='warning')
        self.btn_randomize.on_click(self._randomize_plate)
        
        actions_box = widgets.VBox([
             widgets.HTML("<b>3. Tools</b>"),
             widgets.HBox([self.btn_select_all, self.btn_clear_sel]),
             self.btn_randomize,
             self.btn_unassign
        ], layout=widgets.Layout(border='1px solid #ccc', padding='5px', margin='5px 0'))
        
        # --- 5. Export ---
        self.output_path = widgets.Text(value='experiment_metadata.csv', description='File:')
        self.btn_export = widgets.Button(description="Export CSV", button_style='success')
        self.btn_export.on_click(self._export_csv)
        self.output_log = widgets.Output(layout={'height': '80px', 'border': '1px solid black', 'overflow_y': 'scroll'})
        
        export_box = widgets.VBox([
            widgets.HTML("<b>4. Export</b>"),
            widgets.HBox([self.output_path, self.btn_export]),
            self.output_log
        ], layout=widgets.Layout(border='1px solid #ccc', padding='5px', margin='5px 0'))
        
        # Assemble Sidebar
        sidebar = widgets.VBox([
            compound_box,
            assignment_box,
            actions_box,
            export_box
        ], layout=widgets.Layout(width='400px', margin='0 0 0 10px'))
        
        self.main_layout = widgets.HBox([self.plate_grid, sidebar])
        
    def display(self):
        display(self.main_layout)
        
    # --- Logic ---
    
    def _add_compound_logic(self, name, meta_dict):
        """Internal logic to add compound, separated from UI event."""
        if not name: return False
        
        # Assign Color
        rgb = self.color_cycle[self.color_idx % len(self.color_cycle)]
        hex_c = mcolors.to_hex(rgb)
        
        if name in self.compounds:
            self.compounds[name]['meta'].update(meta_dict)
            return True
            
        self.compounds[name] = {
            'color': hex_c,
            'meta': meta_dict
        }
        self.color_idx += 1
        return True

    def _add_compound(self, _):
        """UI wrapper for adding compound."""
        name = self.txt_compound_name.value.strip()
        meta_str = self.txt_compound_meta.value.strip()
        
        meta_dict = {}
        if meta_str:
            pairs = meta_str.split(';')
            for p in pairs:
                if '=' in p:
                    k, v = p.split('=', 1)
                    meta_dict[k.strip()] = v.strip()
                    
        if self._add_compound_logic(name, meta_dict):
            # Update UI
            self.txt_compound_name.value = ""
            self.txt_compound_meta.value = ""
            self._update_compound_list()
        else:
            with self.output_log: print("Error adding compound: Name required.")

    def _import_csv_lib(self, _):
        path = self.txt_import_path.value
        if not os.path.exists(path):
            with self.output_log: print(f"File not found: {path}")
            return
            
        try:
            df = pd.read_csv(path)
            # Expect 'CompoundName' column
            # If not found, see if we can guess (First column?)
            col_name = None
            for c in df.columns:
                if 'compound' in c.lower() and 'name' in c.lower():
                    col_name = c
                    break
            
            if not col_name:
                col_name = df.columns[0] # Default to 1st
            
            count = 0
            for idx, row in df.iterrows():
                name = str(row[col_name]).strip()
                # Meta is everything else
                meta = {}
                for c in df.columns:
                    if c != col_name:
                        meta[c] = str(row[c])
                
                if self._add_compound_logic(name, meta):
                    count += 1
            
            self._update_compound_list()
            with self.output_log: print(f"Imported {count} compounds from {path}")
            
        except Exception as e:
            with self.output_log: print(f"Import Error: {e}")

    def _update_compound_list(self):
        opts = sorted(list(self.compounds.keys()))
        self.dropdown_compounds.options = opts
        if opts:
            self.dropdown_compounds.value = opts[-1] # Auto-select new
            
        # Legend (HTML)
        html = "<div style='font-size:10px;'>"
        for name, data in self.compounds.items():
            col = data['color']
            extras = ""
            if data['meta']:
                extras = f" <i style='color:#666'>({', '.join([f'{k}={v}' for k,v in data['meta'].items()])})</i>"
            html += f"<span style='color:{col};'>●</span> <b>{name}</b>{extras}<br>"
        html += "</div>"
        with self.compound_legend:
            self.compound_legend.clear_output()
            display(widgets.HTML(html))
            
    def _on_compound_select(self, change):
        self.active_compound = change['new']
        
    def _assign_manual(self, _):
        if not self.active_compound:
             with self.output_log: print("Error: No active compound selected.")
             return
        
        dose = self.input_manual_dose.value
        comp_data = self.compounds[self.active_compound]
        extra_meta = comp_data['meta']
        
        count = 0
        for wid in self.selected_wells:
            if wid not in self.well_data: self.well_data[wid] = {}
            
            # 1. Assign Main Data
            self.well_data[wid][self.compound_key] = self.active_compound
            self.well_data[wid][self.dose_key] = dose
            
            # 2. Assign Fixed Meta
            self.well_data[wid].update(extra_meta)
            
            count += 1
            
        self._refresh_visuals()
        with self.output_log: print(f"Assigned {self.active_compound} to {count} wells.")

    def _generate_dilution(self, _):
        """Fills wells starting from the single selected well."""
        if not self.active_compound:
             with self.output_log: print("Error: No compound selected.")
             return
             
        comp_data = self.compounds[self.active_compound]
        extra_meta = comp_data['meta']
        
        # Get start well
        targets = sorted(list(self.selected_wells))
        if len(targets) != 1:
             with self.output_log: print("Error: Select EXACTLY ONE start well for dilution.")
             return
        
        start_wid = targets[0]
        start_row_idx = self.row_labels.index(start_wid[0])
        start_col_idx = int(start_wid[1:]) - 1
        
        # Params
        dose = self.dil_start.value
        factor = self.dil_factor.value
        steps = self.dil_steps.value
        reps = self.dil_reps.value
        direction = self.dil_dir.value
        
        # Generate
        count = 0
        current_dose = dose
        
        for s in range(steps):
            for r in range(reps):
                r_off = 0
                c_off = 0
                if direction == 'Horizontal': c_off = count
                else: r_off = count
                
                r_idx = start_row_idx + r_off
                c_idx = start_col_idx + c_off
                
                if r_idx < self.rows and c_idx < self.cols:
                    r_lbl = self.row_labels[r_idx]
                    c_lbl = self.col_labels[c_idx]
                    wid = f"{r_lbl}{int(c_lbl):02d}"
                    
                    if wid not in self.well_data: self.well_data[wid] = {}
                    
                    # 1. Assign Main Data
                    self.well_data[wid][self.compound_key] = self.active_compound
                    self.well_data[wid][self.dose_key] = current_dose
                    
                    # 2. Assign Fixed Meta
                    self.well_data[wid].update(extra_meta)
                
                count += 1
            
            # Dilute after reps are done for this step
            current_dose /= factor
            
        self._refresh_visuals()
        with self.output_log: print(f"Generated dilution for {self.active_compound} starting at {start_wid}")

    def _randomize_plate(self, _):
        """Shuffles assigned data globally among the used wells."""
        used_wells = [w for w, d in self.well_data.items() if d]
        if not used_wells: return
        
        data_payloads = [self.well_data[w].copy() for w in used_wells]
        random.shuffle(data_payloads)
        
        for i, wid in enumerate(used_wells):
            self.well_data[wid] = data_payloads[i]
            
        self._refresh_visuals()
        with self.output_log: print(f"Randomized {len(used_wells)} data points.")

    def _unassign_data(self, _):
        """Clears data from selected wells."""
        if not self.selected_wells: return
        
        count = 0
        for wid in self.selected_wells:
            if wid in self.well_data:
                del self.well_data[wid]
                count += 1
        
        self._refresh_visuals()
        with self.output_log: print(f"Unassigned/Cleared data from {count} wells.")

    def _refresh_visuals(self):
        doses = []
        for d in self.well_data.values():
            if self.dose_key in d:
                try: doses.append(float(d[self.dose_key]))
                except: pass
        
        min_d, max_d = (min(doses), max(doses)) if doses else (0, 1)
        
        for wid, btn in self.well_widgets.items():
            is_sel = wid in self.selected_wells
            color = 'lightgray'
            
            if wid in self.well_data:
                d = self.well_data[wid]
                drug = d.get(self.compound_key)
                # Structure changed: self.compounds[drug] is now a dict
                if drug in self.compounds:
                    base = self.compounds[drug]['color'] 
                    
                    # Opacity
                    alpha = 1.0
                    if self.dose_key in d and doses:
                        val = float(d[self.dose_key])
                        rng = max_d - min_d
                        if rng > 0:
                            norm = (val - min_d) / rng
                            alpha = 0.3 + (0.7 * norm)
                    
                    rgb = mcolors.to_rgb(base)
                    blended = tuple(alpha * c + (1-alpha)*1.0 for c in rgb)
                    color = mcolors.to_hex(blended)
            
            if is_sel:
                color = 'orange'
            
            btn.style.button_color = color

    def _on_well_click(self, btn):
        wid = btn.well_id
        if wid in self.selected_wells: self.selected_wells.remove(wid)
        else: self.selected_wells.add(wid)
        self._refresh_visuals()

    def _select_all(self, _):
        self.selected_wells = set(self.well_widgets.keys())
        self._refresh_visuals()
        
    def _clear_selection(self, _):
        self.selected_wells = set()
        self._refresh_visuals()

    def _export_csv(self, _):
        rows = [{'WellID': w, **d} for w, d in self.well_data.items()]
        df = pd.DataFrame(rows)
        try:
            df.to_csv(self.output_path.value, index=False)
            with self.output_log: print(f"Exported to {self.output_path.value}")
        except Exception as e:
            with self.output_log: print(f"Export Error: {e}")
