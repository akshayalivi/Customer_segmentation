import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import io

class CustomerSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f5')
        
        # Initialize data
        self.data = None
        self.clustering_data = None
        self.clusters = None
        self.kms = None
        self.wcss = None
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        self.create_header()
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_content_area()
        
        # Initialize with default data if available
        try:
            self.load_data("Mall_Customers.csv")  # Default dataset
        except:
            pass
    
    def create_header(self):
        header_frame = ttk.Frame(self.main_frame, style='Header.TFrame')
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Logo and title
        logo_img = Image.new('RGB', (40, 40), color='#4a6baf')
        logo_img = ImageTk.PhotoImage(logo_img)
        logo_label = ttk.Label(header_frame, image=logo_img)
        logo_label.image = logo_img
        logo_label.pack(side=tk.LEFT, padx=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="Customer Segmentation Dashboard", 
            font=('Helvetica', 16, 'bold'),
            style='Header.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # Upload button
        upload_btn = ttk.Button(
            header_frame, 
            text="Upload CSV", 
            command=self.upload_file,
            style='Accent.TButton'
        )
        upload_btn.pack(side=tk.RIGHT)
    
    def create_sidebar(self):
        sidebar_frame = ttk.Frame(self.main_frame, width=200, style='Sidebar.TFrame')
        sidebar_frame.pack(fill=tk.Y, side=tk.LEFT, padx=10, pady=10)
        
        # Navigation buttons
        nav_options = [
            ("Overview", self.show_overview),
            ("Demographics", self.show_demographics),
            ("Income Analysis", self.show_income_analysis),
            ("Spending Analysis", self.show_spending_analysis),
            ("Clustering", self.show_clustering),
            ("Full Report", self.show_full_report)
        ]
        
        for text, command in nav_options:
            btn = ttk.Button(
                sidebar_frame, 
                text=text, 
                command=command,
                style='Nav.TButton'
            )
            btn.pack(fill=tk.X, pady=5)
        
        # Data info panel
        data_info_frame = ttk.LabelFrame(sidebar_frame, text="Dataset Info", style='Info.TLabelframe')
        data_info_frame.pack(fill=tk.X, pady=20)
        
        self.data_info_label = ttk.Label(
            data_info_frame, 
            text="No data loaded", 
            style='Info.TLabel',
            wraplength=180
        )
        self.data_info_label.pack(padx=5, pady=5)
    
    def create_content_area(self):
        self.content_frame = ttk.Frame(self.main_frame, style='Content.TFrame')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tabs = {
            "overview": ttk.Frame(self.notebook),
            "demographics": ttk.Frame(self.notebook),
            "income": ttk.Frame(self.notebook),
            "spending": ttk.Frame(self.notebook),
            "clustering": ttk.Frame(self.notebook),
            "report": ttk.Frame(self.notebook)
        }
        
        for name, frame in self.tabs.items():
            self.notebook.add(frame, text=name.capitalize())
        
        # Initially show overview
        self.show_overview()
    
    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            
            # Basic data processing
            if 'CustomerID' in self.data.columns:
                self.data = self.data.drop('CustomerID', axis=1)
            
            # Update data info
            info_text = f"Dataset: {file_path.split('/')[-1]}\n"
            info_text += f"Records: {len(self.data)}\n"
            info_text += f"Columns: {', '.join(self.data.columns)}"
            self.data_info_label.config(text=info_text)
            
            # Prepare clustering data
            if 'Annual Income (k$)' in self.data.columns and 'Spending Score (1-100)' in self.data.columns:
                self.clustering_data = self.data[['Annual Income (k$)', 'Spending Score (1-100)']]
                
                # Perform clustering
                self.perform_clustering()
            
            # Update all visualizations
            self.update_all_visualizations()
            
            # Show success message
            self.show_message("Data loaded successfully!", "success")
            
        except Exception as e:
            self.show_message(f"Error loading file: {str(e)}", "error")
    
    def perform_clustering(self):
        # Elbow method to find optimal k
        self.wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42)
            km.fit(self.clustering_data)
            self.wcss.append(km.inertia_)
        
        # Final clustering with k=5
        self.kms = KMeans(n_clusters=5, init='k-means++', random_state=42)
        self.clusters = self.clustering_data.copy()
        self.clusters['Cluster_Prediction'] = self.kms.fit_predict(self.clustering_data)
    
    def update_all_visualizations(self):
        self.create_overview_tab()
        self.create_demographics_tab()
        self.create_income_analysis_tab()
        self.create_spending_analysis_tab()
        self.create_clustering_tab()
        self.create_full_report_tab()
    
    def create_overview_tab(self):
        # Clear the tab
        for widget in self.tabs["overview"].winfo_children():
            widget.destroy()
        
        # Create overview content
        overview_frame = ttk.Frame(self.tabs["overview"])
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data summary
        summary_frame = ttk.LabelFrame(overview_frame, text="Dataset Summary")
        summary_frame.pack(fill=tk.X, pady=5)
        
        if self.data is not None:
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            info_text = buffer.getvalue()
            
            summary_text = ttk.Label(
                summary_frame, 
                text=info_text,
                font=('Courier', 10),
                justify=tk.LEFT
            )
            summary_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Correlation heatmap
        if self.data is not None and len(self.data.select_dtypes(include=['number']).columns) > 0:
            heatmap_frame = ttk.LabelFrame(overview_frame, text="Correlation Heatmap")
            heatmap_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = self.data.select_dtypes(include=['number']).corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Feature Correlation')
            
            canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_demographics_tab(self):
        # Clear the tab
        for widget in self.tabs["demographics"].winfo_children():
            widget.destroy()
        
        if self.data is None or 'Gender' not in self.data.columns or 'Age' not in self.data.columns:
            return
        
        # Create demographics content
        demographics_frame = ttk.Frame(self.tabs["demographics"])
        demographics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gender distribution
        gender_frame = ttk.LabelFrame(demographics_frame, text="Gender Distribution")
        gender_frame.pack(fill=tk.BOTH, pady=5)
        
        labels = self.data['Gender'].unique()
        values = self.data['Gender'].value_counts(ascending=True)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
        bar = ax0.bar(x=labels, height=values, width=0.4, align='center', color=['#42a7f5','#d400ad'])
        ax0.set(title='Count difference in Gender Distribution', xlabel='Gender', ylabel='No. of Customers')
        ax0.set_ylim(0,130)
        ax0.axhline(y=self.data['Gender'].value_counts()[0], color='#d400ad', linestyle='--', 
                   label=f'Female ({self.data.Gender.value_counts()[0]})')
        ax0.axhline(y=self.data['Gender'].value_counts()[1], color='#42a7f5', linestyle='--', 
                   label=f'Male ({self.data.Gender.value_counts()[1]})')
        ax0.legend()

        ax1.pie(values, labels=labels, colors=['#42a7f5','#d400ad'], autopct='%1.1f%%')
        ax1.set(title='Ratio of Gender Distribution')
        fig.suptitle('Gender Distribution', fontsize=12)
        
        canvas = FigureCanvasTkAgg(fig, master=gender_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Age distribution
        age_frame = ttk.LabelFrame(demographics_frame, text="Age Distribution")
        age_frame.pack(fill=tk.BOTH, pady=5)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(y=self.data["Age"], color="#f73434", ax=ax0)
        ax0.axhline(y=self.data['Age'].max(), linestyle='--',color='#c90404', 
                   label=f'Max Age ({self.data.Age.max()})')
        ax0.axhline(y=self.data['Age'].describe()[6], linestyle='--',color='#f74343', 
                   label=f'75% Age ({self.data.Age.describe()[6]:.2f})')
        ax0.axhline(y=self.data['Age'].median(), linestyle='--',color='#eb50db', 
                   label=f'Median Age ({self.data.Age.median():.2f})')
        ax0.axhline(y=self.data['Age'].describe()[4], linestyle='--',color='#eb50db', 
                   label=f'25% Age ({self.data.Age.describe()[4]:.2f})')
        ax0.axhline(y=self.data['Age'].min(), linestyle='--',color='#046ebf', 
                   label=f'Min Age ({self.data.Age.min()})')
        ax0.legend(fontsize='xx-small', loc='upper right')
        ax0.set_ylabel('No. of Customers')
        ax0.set_title('Age Distribution')

        # Histogram
        sns.histplot(self.data['Age'], bins=15, ax=ax1, color='orange')
        ax1.set_xlabel('Age')
        ax1.set_title('Age Distribution Histogram')
        
        canvas = FigureCanvasTkAgg(fig, master=age_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_income_analysis_tab(self):
        # Clear the tab
        for widget in self.tabs["income"].winfo_children():
            widget.destroy()
        
        if self.data is None or 'Annual Income (k$)' not in self.data.columns:
            return
        
        # Create income analysis content
        income_frame = ttk.Frame(self.tabs["income"])
        income_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Income distribution
        dist_frame = ttk.LabelFrame(income_frame, text="Income Distribution")
        dist_frame.pack(fill=tk.BOTH, pady=5)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(y=self.data["Annual Income (k$)"], color="#f73434", ax=ax0)
        ax0.axhline(y=self.data["Annual Income (k$)"].max(), linestyle='--',color='#c90404', 
                   label=f'Max Income ({self.data["Annual Income (k$)"].max()})')
        ax0.axhline(y=self.data["Annual Income (k$)"].describe()[6], linestyle='--',color='#f74343', 
                   label=f'75% Income ({self.data["Annual Income (k$)"].describe()[6]:.2f})')
        ax0.axhline(y=self.data["Annual Income (k$)"].median(), linestyle='--',color='#eb50db', 
                   label=f'Median Income ({self.data["Annual Income (k$)"].median():.2f})')
        ax0.axhline(y=self.data["Annual Income (k$)"].describe()[4], linestyle='--',color='#eb50db', 
                   label=f'25% Income ({self.data["Annual Income (k$)"].describe()[4]:.2f})')
        ax0.axhline(y=self.data["Annual Income (k$)"].min(), linestyle='--',color='#046ebf', 
                   label=f'Min Income ({self.data["Annual Income (k$)"].min()})')
        ax0.legend(fontsize='xx-small', loc='upper right')
        ax0.set_ylabel('Annual Income (k$)')
        ax0.set_title('Income Distribution')

        # Histogram
        sns.histplot(self.data['Annual Income (k$)'], bins=15, ax=ax1, color='orange')
        ax1.set_xlabel('Annual Income (k$)')
        ax1.set_title('Income Distribution Histogram')
        
        canvas = FigureCanvasTkAgg(fig, master=dist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Income by gender
        gender_frame = ttk.LabelFrame(income_frame, text="Income by Gender")
        gender_frame.pack(fill=tk.BOTH, pady=5)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(x=self.data['Gender'], y=self.data["Annual Income (k$)"], 
                   hue=self.data['Gender'], palette='seismic', ax=ax0)
        ax0.set_ylabel('Annual Income (k$)')
        ax0.set_title('Income Distribution by Gender')

        # Violin plot
        sns.violinplot(y=self.data['Annual Income (k$)'], x=self.data['Gender'], ax=ax1)
        ax1.set_ylabel('Annual Income (k$)')
        ax1.set_title('Income Distribution by Gender')
        
        canvas = FigureCanvasTkAgg(fig, master=gender_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_spending_analysis_tab(self):
        # Clear the tab
        for widget in self.tabs["spending"].winfo_children():
            widget.destroy()
        
        if self.data is None or 'Spending Score (1-100)' not in self.data.columns:
            return
        
        # Create spending analysis content
        spending_frame = ttk.Frame(self.tabs["spending"])
        spending_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Spending distribution
        dist_frame = ttk.LabelFrame(spending_frame, text="Spending Score Distribution")
        dist_frame.pack(fill=tk.BOTH, pady=5)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(y=self.data["Spending Score (1-100)"], color="#f73434", ax=ax0)
        ax0.axhline(y=self.data["Spending Score (1-100)"].max(), linestyle='--',color='#c90404', 
                   label=f'Max Spending ({self.data["Spending Score (1-100)"].max()})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].describe()[6], linestyle='--',color='#f74343', 
                   label=f'75% Spending ({self.data["Spending Score (1-100)"].describe()[6]:.2f})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].median(), linestyle='--',color='#eb50db', 
                   label=f'Median Spending ({self.data["Spending Score (1-100)"].median():.2f})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].describe()[4], linestyle='--',color='#eb50db', 
                   label=f'25% Spending ({self.data["Spending Score (1-100)"].describe()[4]:.2f})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].min(), linestyle='--',color='#046ebf', 
                   label=f'Min Spending ({self.data["Spending Score (1-100)"].min()})')
        ax0.legend(fontsize='xx-small', loc='upper right')
        ax0.set_ylabel('Spending Score')
        ax0.set_title('Spending Score Distribution')

        # Histogram
        sns.histplot(self.data['Spending Score (1-100)'], bins=15, ax=ax1, color='orange')
        ax1.set_xlabel('Spending Score (1-100)')
        ax1.set_title('Spending Score Histogram')
        
        canvas = FigureCanvasTkAgg(fig, master=dist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Spending by gender
        gender_frame = ttk.LabelFrame(spending_frame, text="Spending Score by Gender")
        gender_frame.pack(fill=tk.BOTH, pady=5)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(x=self.data['Gender'], y=self.data["Spending Score (1-100)"], 
                   hue=self.data['Gender'], palette='seismic', ax=ax0)
        ax0.set_ylabel('Spending Score')
        ax0.set_title('Spending Score by Gender')

        # Violin plot
        sns.violinplot(y=self.data['Spending Score (1-100)'], x=self.data['Gender'], ax=ax1)
        ax1.set_ylabel('Spending Score')
        ax1.set_title('Spending Score by Gender')
        
        canvas = FigureCanvasTkAgg(fig, master=gender_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_clustering_tab(self):
        # Clear the tab
        for widget in self.tabs["clustering"].winfo_children():
            widget.destroy()
        
        if self.clusters is None or self.kms is None or self.wcss is None:
            return
        
        # Create clustering content
        clustering_frame = ttk.Frame(self.tabs["clustering"])
        clustering_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for clustering visualizations
        cluster_notebook = ttk.Notebook(clustering_frame)
        cluster_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab for main clusters visualization
        main_cluster_frame = ttk.Frame(cluster_notebook)
        cluster_notebook.add(main_cluster_frame, text="Customer Segments")
        
        # Tab for elbow method
        elbow_frame = ttk.Frame(cluster_notebook)
        cluster_notebook.add(elbow_frame, text="Elbow Method")
        
        # Main clusters visualization
        main_vis_frame = ttk.LabelFrame(main_cluster_frame, text="Customer Segments")
        main_vis_frame.pack(fill=tk.BOTH, pady=5, padx=5, expand=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot clusters
        colors = ['orange', 'deepskyblue', 'magenta', 'red', 'lime']
        for i in range(5):
            cluster_data = self.clusters[self.clusters['Cluster_Prediction'] == i]
            ax.scatter(
                x=cluster_data['Annual Income (k$)'],
                y=cluster_data['Spending Score (1-100)'],
                s=70, edgecolor='black', linewidth=0.3, 
                c=colors[i], label=f'Cluster {i+1}'
            )
        
        # Plot centroids
        ax.scatter(
            x=self.kms.cluster_centers_[:, 0], 
            y=self.kms.cluster_centers_[:, 1], 
            s=120, c='yellow', label='Centroids',
            edgecolor='black', linewidth=0.3
        )
        
        ax.legend(loc='upper right')
        ax.set_xlim(0, 140)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Annual Income (in Thousand USD)')
        ax.set_ylabel('Spending Score')
        ax.set_title('Customer Segments by Income and Spending')
        
        canvas = FigureCanvasTkAgg(fig, master=main_vis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Elbow method visualization
        elbow_vis_frame = ttk.LabelFrame(elbow_frame, text="Elbow Method for Optimal K")
        elbow_vis_frame.pack(fill=tk.BOTH, pady=5, padx=5, expand=True)
        
        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(range(1, 11), self.wcss, linewidth=2, color="red", marker="8")
        ax_elbow.axvline(x=5, ls='--', color='blue', label='Optimal K (5)')
        ax_elbow.set_title('The Elbow Method')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        ax_elbow.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
        ax_elbow.legend()
        ax_elbow.grid(True)
        
        canvas_elbow = FigureCanvasTkAgg(fig_elbow, master=elbow_vis_frame)
        canvas_elbow.draw()
        canvas_elbow.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Cluster details
        details_frame = ttk.LabelFrame(main_cluster_frame, text="Cluster Details")
        details_frame.pack(fill=tk.BOTH, pady=5)
        
        # Create a frame for cluster statistics
        stats_frame = ttk.Frame(details_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Add cluster statistics
        for i in range(5):
            cluster_data = self.clusters[self.clusters['Cluster_Prediction'] == i]
            cluster_stats = f"Cluster {i+1}:\n"
            cluster_stats += f"Size: {len(cluster_data)} customers\n"
            cluster_stats += f"Avg Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k\n"
            cluster_stats += f"Avg Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}"
            
            cluster_label = ttk.Label(
                stats_frame, 
                text=cluster_stats,
                relief=tk.RIDGE,
                padding=10,
                background=colors[i],
                foreground='white' if i in [2, 3] else 'black',
                font=('Helvetica', 10, 'bold')
            )
            cluster_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def create_full_report_tab(self):
        # Clear the tab
        for widget in self.tabs["report"].winfo_children():
            widget.destroy()
        
        if self.data is None:
            return
        
        # Create report content
        report_frame = ttk.Frame(self.tabs["report"])
        report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(report_frame)
        scrollbar = ttk.Scrollbar(report_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add all visualizations to the report
        self.add_report_section(scrollable_frame, "Gender Distribution", self.create_gender_plot)
        self.add_report_section(scrollable_frame, "Age Distribution", self.create_age_plot)
        self.add_report_section(scrollable_frame, "Income Distribution", self.create_income_plot)
        self.add_report_section(scrollable_frame, "Spending Score Distribution", self.create_spending_plot)
        
        if self.clusters is not None:
            self.add_report_section(scrollable_frame, "Customer Segments", self.create_cluster_plot)
            self.add_report_section(scrollable_frame, "Elbow Method", self.create_elbow_plot)
    
    def add_report_section(self, parent, title, plot_function):
        section_frame = ttk.LabelFrame(parent, text=title)
        section_frame.pack(fill=tk.X, pady=10, padx=5)
        
        fig = plot_function()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=section_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_gender_plot(self):
        if 'Gender' not in self.data.columns:
            return None
            
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
        labels = self.data['Gender'].unique()
        values = self.data['Gender'].value_counts(ascending=True)
        
        bar = ax0.bar(x=labels, height=values, width=0.4, align='center', color=['#42a7f5','#d400ad'])
        ax0.set(title='Count difference in Gender Distribution', xlabel='Gender', ylabel='No. of Customers')
        ax0.set_ylim(0,130)
        ax0.axhline(y=self.data['Gender'].value_counts()[0], color='#d400ad', linestyle='--', 
                   label=f'Female ({self.data.Gender.value_counts()[0]})')
        ax0.axhline(y=self.data['Gender'].value_counts()[1], color='#42a7f5', linestyle='--', 
                   label=f'Male ({self.data.Gender.value_counts()[1]})')
        ax0.legend()

        ax1.pie(values, labels=labels, colors=['#42a7f5','#d400ad'], autopct='%1.1f%%')
        ax1.set(title='Ratio of Gender Distribution')
        fig.suptitle('Gender Distribution', fontsize=12)
        
        return fig
    
    def create_age_plot(self):
        if 'Age' not in self.data.columns:
            return None
            
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(y=self.data["Age"], color="#f73434", ax=ax0)
        ax0.axhline(y=self.data['Age'].max(), linestyle='--',color='#c90404', 
                   label=f'Max Age ({self.data.Age.max()})')
        ax0.axhline(y=self.data['Age'].describe()[6], linestyle='--',color='#f74343', 
                   label=f'75% Age ({self.data.Age.describe()[6]:.2f})')
        ax0.axhline(y=self.data['Age'].median(), linestyle='--',color='#eb50db', 
                   label=f'Median Age ({self.data.Age.median():.2f})')
        ax0.axhline(y=self.data['Age'].describe()[4], linestyle='--',color='#eb50db', 
                   label=f'25% Age ({self.data.Age.describe()[4]:.2f})')
        ax0.axhline(y=self.data['Age'].min(), linestyle='--',color='#046ebf', 
                   label=f'Min Age ({self.data.Age.min()})')
        ax0.legend(fontsize='xx-small', loc='upper right')
        ax0.set_ylabel('No. of Customers')
        ax0.set_title('Age Distribution')

        # Histogram
        sns.histplot(self.data['Age'], bins=15, ax=ax1, color='orange')
        ax1.set_xlabel('Age')
        ax1.set_title('Age Distribution Histogram')
        
        return fig
    
    def create_income_plot(self):
        if 'Annual Income (k$)' not in self.data.columns:
            return None
            
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(y=self.data["Annual Income (k$)"], color="#f73434", ax=ax0)
        ax0.axhline(y=self.data["Annual Income (k$)"].max(), linestyle='--',color='#c90404', 
                   label=f'Max Income ({self.data["Annual Income (k$)"].max()})')
        ax0.axhline(y=self.data["Annual Income (k$)"].describe()[6], linestyle='--',color='#f74343', 
                   label=f'75% Income ({self.data["Annual Income (k$)"].describe()[6]:.2f})')
        ax0.axhline(y=self.data["Annual Income (k$)"].median(), linestyle='--',color='#eb50db', 
                   label=f'Median Income ({self.data["Annual Income (k$)"].median():.2f})')
        ax0.axhline(y=self.data["Annual Income (k$)"].describe()[4], linestyle='--',color='#eb50db', 
                   label=f'25% Income ({self.data["Annual Income (k$)"].describe()[4]:.2f})')
        ax0.axhline(y=self.data["Annual Income (k$)"].min(), linestyle='--',color='#046ebf', 
                   label=f'Min Income ({self.data["Annual Income (k$)"].min()})')
        ax0.legend(fontsize='xx-small', loc='upper right')
        ax0.set_ylabel('Annual Income (k$)')
        ax0.set_title('Income Distribution')

        # Histogram
        sns.histplot(self.data['Annual Income (k$)'], bins=15, ax=ax1, color='orange')
        ax1.set_xlabel('Annual Income (k$)')
        ax1.set_title('Income Distribution Histogram')
        
        return fig
    
    def create_spending_plot(self):
        if 'Spending Score (1-100)' not in self.data.columns:
            return None
            
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4))
        
        # Boxplot
        sns.boxplot(y=self.data["Spending Score (1-100)"], color="#f73434", ax=ax0)
        ax0.axhline(y=self.data["Spending Score (1-100)"].max(), linestyle='--',color='#c90404', 
                   label=f'Max Spending ({self.data["Spending Score (1-100)"].max()})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].describe()[6], linestyle='--',color='#f74343', 
                   label=f'75% Spending ({self.data["Spending Score (1-100)"].describe()[6]:.2f})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].median(), linestyle='--',color='#eb50db', 
                   label=f'Median Spending ({self.data["Spending Score (1-100)"].median():.2f})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].describe()[4], linestyle='--',color='#eb50db', 
                   label=f'25% Spending ({self.data["Spending Score (1-100)"].describe()[4]:.2f})')
        ax0.axhline(y=self.data["Spending Score (1-100)"].min(), linestyle='--',color='#046ebf', 
                   label=f'Min Spending ({self.data["Spending Score (1-100)"].min()})')
        ax0.legend(fontsize='xx-small', loc='upper right')
        ax0.set_ylabel('Spending Score')
        ax0.set_title('Spending Score Distribution')

        # Histogram
        sns.histplot(self.data['Spending Score (1-100)'], bins=15, ax=ax1, color='orange')
        ax1.set_xlabel('Spending Score (1-100)')
        ax1.set_title('Spending Score Histogram')
        
        return fig
    
    def create_cluster_plot(self):
        if self.clusters is None or self.kms is None:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot clusters
        colors = ['orange', 'deepskyblue', 'magenta', 'red', 'lime']
        for i in range(5):
            cluster_data = self.clusters[self.clusters['Cluster_Prediction'] == i]
            ax.scatter(
                x=cluster_data['Annual Income (k$)'],
                y=cluster_data['Spending Score (1-100)'],
                s=70, edgecolor='black', linewidth=0.3, 
                c=colors[i], label=f'Cluster {i+1}'
            )
        
        # Plot centroids
        ax.scatter(
            x=self.kms.cluster_centers_[:, 0], 
            y=self.kms.cluster_centers_[:, 1], 
            s=120, c='yellow', label='Centroids',
            edgecolor='black', linewidth=0.3
        )
        
        ax.legend(loc='upper right')
        ax.set_xlim(0, 140)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Annual Income (in Thousand USD)')
        ax.set_ylabel('Spending Score')
        ax.set_title('Customer Segments by Income and Spending')
        
        return fig
    
    def create_elbow_plot(self):
        if self.wcss is None:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, 11), self.wcss, linewidth=2, color="red", marker="8")
        ax.axvline(x=5, ls='--', color='blue', label='Optimal K (5)')
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def show_overview(self):
        self.notebook.select(self.tabs["overview"])
    
    def show_demographics(self):
        self.notebook.select(self.tabs["demographics"])
    
    def show_income_analysis(self):
        self.notebook.select(self.tabs["income"])
    
    def show_spending_analysis(self):
        self.notebook.select(self.tabs["spending"])
    
    def show_clustering(self):
        self.notebook.select(self.tabs["clustering"])
    
    def show_full_report(self):
        self.notebook.select(self.tabs["report"])
    
    def show_message(self, message, message_type):
        # Create a temporary message label
        message_frame = ttk.Frame(self.main_frame, style='Message.TFrame')
        message_frame.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
        
        bg_color = '#4CAF50' if message_type == "success" else '#F44336'
        message_label = ttk.Label(
            message_frame, 
            text=message, 
            style='Message.TLabel',
            background=bg_color,
            foreground='white',
            padding=10
        )
        message_label.pack()
        
        # Remove the message after 3 seconds
        self.root.after(3000, message_frame.destroy)

# Create the main window
root = tk.Tk()

# Configure styles
style = ttk.Style()
style.configure('Header.TFrame', background='#4a6baf')
style.configure('Header.TLabel', background='#4a6baf', foreground='white')
style.configure('Sidebar.TFrame', background='#e0e0e0', relief=tk.RAISED, borderwidth=1)
style.configure('Content.TFrame', background='#ffffff')
style.configure('Info.TLabelframe', background='#e0e0e0')
style.configure('Info.TLabel', background='#e0e0e0')
style.configure('Accent.TButton', background='#4a6baf', foreground='white', font=('Helvetica', 10, 'bold'))
style.configure('Nav.TButton', width=20, font=('Helvetica', 10))
style.configure('Message.TFrame', background='#4CAF50')
style.configure('Message.TLabel', background='#4CAF50', foreground='white', font=('Helvetica', 10, 'bold'))

# Run the application
app = CustomerSegmentationApp(root)
root.mainloop()