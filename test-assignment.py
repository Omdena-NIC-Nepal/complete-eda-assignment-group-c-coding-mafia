import unittest
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import re
import pandas as pd
import numpy as np

class TestClimateEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the notebook
        with open('climate_eda.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = nbformat.read(f, as_version=4)
        
        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(cls.notebook, {'metadata': {'path': '.'}})
        
        # Extract code and markdown cells
        cls.code_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'code']
        cls.markdown_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'markdown']
        cls.all_code = '\n'.join([cell['source'] for cell in cls.code_cells])
        cls.all_markdown = '\n'.join([cell['source'] for cell in cls.markdown_cells])
        
        # Check if data was loaded properly
        for cell in cls.code_cells:
            if 'df = pd.read_csv' in cell['source']:
                # Get variable name of dataframe
                match = re.search(r'(\w+)\s*=\s*pd\.read_csv', cell['source'])
                if match:
                    cls.df_name = match.group(1)
                    break
        
    def test_required_libraries(self):
        """Test that all required libraries are imported"""
        required_libs = ['pandas', 'numpy', 'matplotlib', 'seaborn']
        for lib in required_libs:
            self.assertIn(f"import {lib}", self.all_code, f"Missing required import for {lib}")
            
    def test_data_loading(self):
        """Test that climate data is loaded"""
        self.assertIn("read_csv('data/Climate_Change_Indicators.csv')", self.all_code, "Data file not loaded correctly")
        
    def test_yearly_aggregation(self):
        """Test that data is aggregated by year"""
        # Look for groupby operations on year
        yearly_agg_patterns = [
            r"groupby\(\s*['\"]Year['\"]\s*\)",
            r"groupby\(\s*['\"]\w+['\"]\s*\)\[['\"]\w+['\"]",
            r"resample\(\s*['\"]Y['\"]\s*\)"
        ]
        found_yearly_agg = any(re.search(pattern, self.all_code) for pattern in yearly_agg_patterns)
        self.assertTrue(found_yearly_agg, "No evidence of yearly data aggregation")
        
    def test_univariate_analysis(self):
        """Test for univariate analysis visualizations and statistics"""
        univariate_vis_patterns = [
            r"hist(plot)?\(",
            r"boxplot\(",
            r"plot\(",
            r"displot\(",
            r"kdeplot\("
        ]
        found_univariate_vis = any(re.search(pattern, self.all_code) for pattern in univariate_vis_patterns)
        self.assertTrue(found_univariate_vis, "No evidence of univariate visualizations")
        
        # Check for descriptive statistics
        stats_patterns = [r"describe\(", r"mean\(", r"median\(", r"std\(", r"min\(", r"max\("]
        found_stats = any(re.search(pattern, self.all_code) for pattern in stats_patterns)
        self.assertTrue(found_stats, "No evidence of descriptive statistics calculation")
        
    def test_bivariate_analysis(self):
        """Test for bivariate analysis"""
        bivariate_vis_patterns = [
            r"scatter(plot)?\(",
            r"reg(plot)?\(",
            r"lineplot\(",
            r"barplot\(",
            r"violinplot\(",
            r"heatmap\(",
            r"corr\("
        ]
        found_bivariate_vis = any(re.search(pattern, self.all_code) for pattern in bivariate_vis_patterns)
        self.assertTrue(found_bivariate_vis, "No evidence of bivariate visualizations")
        
        # Check for correlation analysis
        corr_patterns = [r"corr\(", r"corrplot", r"corrcoef"]
        found_corr = any(re.search(pattern, self.all_code) for pattern in corr_patterns)
        self.assertTrue(found_corr, "No evidence of correlation analysis")
        
    def test_multivariate_analysis(self):
        """Test for multivariate analysis"""
        multivariate_vis_patterns = [
            r"pairplot\(",
            r"PCA\(",
            r"heatmap\(",
            r"parallel_coordinates\(",
            r"andrews_curves\(",
            r"radviz\(",
            r"3d scatter"
        ]
        found_multivariate_vis = any(re.search(pattern, self.all_code) for pattern in multivariate_vis_patterns)
        self.assertTrue(found_multivariate_vis, "No evidence of multivariate visualizations")
        
    def test_conclusions_present(self):
        """Test that conclusions are present in markdown cells"""
        conclusion_patterns = [
            r"[Cc]onclusion",
            r"[Ff]inding",
            r"[Ss]ummar",
            r"[Ii]nsight",
            r"[Oo]bservation"
        ]
        found_conclusion = any(re.search(pattern, self.all_markdown) for pattern in conclusion_patterns)
        self.assertTrue(found_conclusion, "No evidence of conclusions or insights in the analysis")
        
    def test_min_number_of_visualizations(self):
        """Test that there are at least 5 different visualizations"""
        vis_function_patterns = [
            r"plt\.\w+\(",
            r"sns\.\w+\(",
            r"df\.\w+\.plot\("
        ]
        num_vis = sum(len(re.findall(pattern, self.all_code)) for pattern in vis_function_patterns)
        self.assertGreaterEqual(num_vis, 5, "Insufficient number of visualizations (minimum 5 required)")
        
    def test_climate_variables_analyzed(self):
        """Test that all climate variables are analyzed"""
        climate_vars = ['Global Average Temperature (°C)', 'CO2 Concentration (ppm)', 'Sea Level Rise (mm)', 'Arctic Ice Area (million km²)']
        for var in climate_vars:
            self.assertIn(var, self.all_code, f"Climate variable {var} not analyzed")

    def calculate_grade(self):
        """Calculate the grade based on passing tests"""
        # List of all test methods
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        total_tests = len(test_methods)
        
        # Run all tests and count passed tests
        passed_tests = 0
        for test in test_methods:
            try:
                getattr(self, test)()
                passed_tests += 1
            except AssertionError:
                continue
        
        # Calculate grade (out of 100)
        grade = (passed_tests / total_tests) * 100
        return round(grade)

if __name__ == '__main__':
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestClimateEDA)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Calculate and print grade
    test_case = TestClimateEDA()
    grade = test_case.calculate_grade()
    print(f"\nFinal Grade: {grade}/100")
