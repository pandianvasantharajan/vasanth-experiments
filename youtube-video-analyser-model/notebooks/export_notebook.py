#!/usr/bin/env python3
"""
Export Jupyter notebook to HTML (which can then be printed to PDF)
"""
import json
from pathlib import Path
from datetime import datetime

def notebook_to_html(notebook_path, output_path=None):
    """Convert notebook to HTML with execution results"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.html')
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>YouTube Video Analyser - Notebook Export</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .notebook-container {
            background: white;
            padding: 40px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .cell {
            margin-bottom: 20px;
            break-inside: avoid;
        }
        .cell-input {
            background: #f8f9fa;
            border-left: 3px solid #007acc;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .cell-output {
            background: #ffffff;
            border-left: 3px solid #28a745;
            padding: 15px;
            margin-top: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
        }
        .markdown-cell {
            padding: 15px 0;
        }
        .markdown-cell h1 {
            color: #1a1a1a;
            border-bottom: 2px solid #e1e4e8;
            padding-bottom: 10px;
        }
        .markdown-cell h2 {
            color: #2a2a2a;
            margin-top: 30px;
        }
        .markdown-cell h3 {
            color: #3a3a3a;
        }
        .markdown-cell code {
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        .code {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
        }
        .execution-count {
            color: #666;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }
        .header h1 {
            color: #007acc;
            margin-bottom: 5px;
        }
        .header .date {
            color: #666;
            font-size: 14px;
        }
        @media print {
            body {
                background: white;
            }
            .notebook-container {
                box-shadow: none;
                padding: 0;
            }
            .cell {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="notebook-container">
        <div class="header">
            <h1>YouTube Video Analyser</h1>
            <p class="date">Exported on """ + datetime.now().strftime("%B %d, %Y at %I:%M %p") + """</p>
        </div>
"""
    
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'markdown':
            html_content += f'<div class="cell markdown-cell">\n'
            # Simple markdown to HTML conversion
            html_source = source
            # Convert headers
            html_source = html_source.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
            html_source = html_source.replace('## ', '<h2>').replace('\n', '</h2>\n')
            html_source = html_source.replace('### ', '<h3>').replace('\n', '</h3>\n')
            # Convert code blocks
            html_source = html_source.replace('`', '<code>').replace('`', '</code>')
            # Convert line breaks
            html_source = html_source.replace('\n\n', '</p><p>').replace('\n', '<br>')
            html_content += f'<p>{html_source}</p>\n'
            html_content += '</div>\n'
            
        elif cell_type == 'code':
            execution_count = cell.get('execution_count', '')
            html_content += '<div class="cell code-cell">\n'
            
            if execution_count:
                html_content += f'<div class="execution-count">In [{execution_count}]:</div>\n'
            
            html_content += '<div class="cell-input code">\n'
            html_content += source.replace('<', '&lt;').replace('>', '&gt;')
            html_content += '\n</div>\n'
            
            # Add outputs
            outputs = cell.get('outputs', [])
            if outputs:
                for output in outputs:
                    output_type = output.get('output_type', '')
                    
                    if output_type == 'stream':
                        text = ''.join(output.get('text', []))
                        html_content += '<div class="cell-output">\n'
                        html_content += text.replace('<', '&lt;').replace('>', '&gt;')
                        html_content += '</div>\n'
                    
                    elif output_type == 'execute_result' or output_type == 'display_data':
                        data = output.get('data', {})
                        if 'text/plain' in data:
                            text = ''.join(data['text/plain'])
                            html_content += '<div class="cell-output">\n'
                            html_content += text.replace('<', '&lt;').replace('>', '&gt;')
                            html_content += '</div>\n'
                    
                    elif output_type == 'error':
                        error_text = '\n'.join(output.get('traceback', []))
                        html_content += '<div class="cell-output" style="border-left-color: #dc3545; color: #dc3545;">\n'
                        html_content += error_text.replace('<', '&lt;').replace('>', '&gt;')
                        html_content += '</div>\n'
            
            html_content += '</div>\n'
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Notebook exported successfully!")
    print(f"  Input:  {notebook_path}")
    print(f"  Output: {output_path}")
    print(f"\nTo convert to PDF:")
    print(f"  1. Open {output_path} in your browser")
    print(f"  2. Press Cmd+P (or Ctrl+P)")
    print(f"  3. Select 'Save as PDF' as the destination")
    print(f"  4. Click 'Save'")

if __name__ == '__main__':
    notebook_file = 'youtube-video-analyser.ipynb'
    notebook_to_html(notebook_file)
