
import os
import json
import anthropic
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import docx
from datetime import datetime
import argparse


class ResumeRubricGenerator:
    """
    Generates a comprehensive rubric for scoring VC+Founders Dinner applications
    using multiple LLMs (Claude and Gemini) for ensemble-based rubric creation.
    """
    
    def __init__(self, resume_dir: str = "resumes"):
        """
        Initialize the rubric generator.
        
        Args:
            resume_dir: Directory containing resume files
        """
        self.resume_dir = Path(resume_dir)
        self.resumes = []
        self.rubric = None
        
        # Initialize API clients
        self.claude_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        text = ""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
        return text
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def load_resumes(self) -> List[Dict[str, Any]]:
        """
        Load all resumes from the resume directory.
        
        Returns:
            List of dictionaries containing resume metadata and content
        """
        print(f"Loading resumes from {self.resume_dir}...")
        
        if not self.resume_dir.exists():
            raise FileNotFoundError(f"Resume directory '{self.resume_dir}' not found")
        
        resume_files = list(self.resume_dir.glob("*"))
        
        for file_path in resume_files:
            if file_path.is_file():
                text = ""
                
                # Extract text based on file type
                if file_path.suffix.lower() == '.pdf':
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.suffix.lower() == '.docx':
                    text = self.extract_text_from_docx(file_path)
                elif file_path.suffix.lower() == '.txt':
                    text = self.extract_text_from_txt(file_path)
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue
                
                if text.strip():
                    self.resumes.append({
                        'filename': file_path.name,
                        'content': text.strip(),
                        'file_path': str(file_path)
                    })
                    print(f"✓ Loaded: {file_path.name}")
        
        print(f"\nTotal resumes loaded: {len(self.resumes)}\n")
        return self.resumes
    
    def create_resume_summary(self) -> str:
        """
        Create a summary of all resumes for rubric generation context.
        
        Returns:
            String summary of resume pool characteristics
        """
        summary_parts = []
        summary_parts.append(f"Total number of candidates: {len(self.resumes)}")
        
        # Sample a few resumes for diversity analysis (first 200 chars of each)
        sample_profiles = []
        for i, resume in enumerate(self.resumes[:10]):  # Sample up to 10 resumes
            preview = resume['content'][:300].replace('\n', ' ')
            sample_profiles.append(f"Candidate {i+1} preview: {preview}...")
        
        summary_parts.append("\nSample candidate profiles:\n" + "\n\n".join(sample_profiles))
        
        return "\n".join(summary_parts)
    
    def generate_rubric_with_claude(self, user_prompt: str, resume_summary: str) -> Dict[str, Any]:
        """
        Generate rubric using Claude.
        
        Args:
            user_prompt: Denny's specific requirements for the dinner
            resume_summary: Summary of the resume pool
            
        Returns:
            Rubric dictionary from Claude
        """
        print("Generating rubric with Claude...")
        
        system_prompt = """You are an expert at creating evaluation rubrics for venture capital and startup founder assessments. 
Your task is to create a comprehensive, granular rubric that can differentiate between candidates based on their resumes."""
        
        user_message = f"""Create a detailed scoring rubric for evaluating candidates for a VC+Founders Dinner event.

**Event Requirements:**
{user_prompt}

**Candidate Pool Context:**
{resume_summary}

**Instructions:**
Create a comprehensive rubric with the following structure:

1. **Crackedness Score (0-100 points)**: Measures overall talent, achievement, and potential
   - Break this down into 5-7 specific evaluation criteria
   - Each criterion should have:
     * Name
     * Description of what it measures
     * Point allocation (totaling 100)
     * Scoring guidelines (what earns different point levels)

2. **Fit Score (0-100 points)**: Measures alignment with the specific event focus area
   - Break this down into 5-7 specific evaluation criteria
   - Each criterion should have:
     * Name
     * Description of what it measures
     * Point allocation (totaling 100)
     * Scoring guidelines (what earns different point levels)

Make the rubric:
- Specific and measurable
- Differentiated (can separate strong from stronger candidates)
- Fair and objective
- Relevant to the candidate pool you see

Return the rubric as a JSON object with this structure:
{{
  "crackedness_criteria": [
    {{
      "name": "criterion name",
      "description": "what this measures",
      "max_points": X,
      "scoring_guide": {{
        "high": "description of what earns 80-100% of points",
        "medium": "description of what earns 40-79% of points",
        "low": "description of what earns 0-39% of points"
      }}
    }}
  ],
  "fit_criteria": [
    {{
      "name": "criterion name",
      "description": "what this measures",
      "max_points": X,
      "scoring_guide": {{
        "high": "description of what earns 80-100% of points",
        "medium": "description of what earns 40-79% of points",
        "low": "description of what earns 0-39% of points"
      }}
    }}
  ],
  "metadata": {{
    "focus_area": "brief description",
    "created_by": "claude"
  }}
}}"""
        
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                rubric_json = response_text[json_start:json_end]
                return json.loads(rubric_json)
            else:
                raise ValueError("No valid JSON found in Claude response")
                
        except Exception as e:
            print(f"Error generating rubric with Claude: {e}")
            return None
    
    def generate_rubric_with_gemini(self, user_prompt: str, resume_summary: str) -> Dict[str, Any]:
        """
        Generate rubric using Gemini.
        
        Args:
            user_prompt: Denny's specific requirements for the dinner
            resume_summary: Summary of the resume pool
            
        Returns:
            Rubric dictionary from Gemini
        """
        print("Generating rubric with Gemini...")
        
        prompt = f"""Create a detailed scoring rubric for evaluating candidates for a VC+Founders Dinner event.

**Event Requirements:**
{user_prompt}

**Candidate Pool Context:**
{resume_summary}

**Instructions:**
Create a comprehensive rubric with the following structure:

1. **Crackedness Score (0-100 points)**: Measures overall talent, achievement, and potential
   - Break this down into 5-7 specific evaluation criteria
   - Each criterion should have:
     * Name
     * Description of what it measures
     * Point allocation (totaling 100)
     * Scoring guidelines (what earns different point levels)

2. **Fit Score (0-100 points)**: Measures alignment with the specific event focus area
   - Break this down into 5-7 specific evaluation criteria
   - Each criterion should have:
     * Name
     * Description of what it measures
     * Point allocation (totaling 100)
     * Scoring guidelines (what earns different point levels)

Make the rubric:
- Specific and measurable
- Differentiated (can separate strong from stronger candidates)
- Fair and objective
- Relevant to the candidate pool you see

Return ONLY a valid JSON object with this structure:
{{
  "crackedness_criteria": [
    {{
      "name": "criterion name",
      "description": "what this measures",
      "max_points": X,
      "scoring_guide": {{
        "high": "description of what earns 80-100% of points",
        "medium": "description of what earns 40-79% of points",
        "low": "description of what earns 0-39% of points"
      }}
    }}
  ],
  "fit_criteria": [
    {{
      "name": "criterion name",
      "description": "what this measures",
      "max_points": X,
      "scoring_guide": {{
        "high": "description of what earns 80-100% of points",
        "medium": "description of what earns 40-79% of points",
        "low": "description of what earns 0-39% of points"
      }}
    }}
  ],
  "metadata": {{
    "focus_area": "brief description",
    "created_by": "gemini"
  }}
}}

Return ONLY the JSON, no other text."""
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000,
                )
            )
            
            response_text = response.text
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                rubric_json = response_text[json_start:json_end]
                return json.loads(rubric_json)
            else:
                raise ValueError("No valid JSON found in Gemini response")
                
        except Exception as e:
            print(f"Error generating rubric with Gemini: {e}")
            return None
    
    def merge_rubrics(self, claude_rubric: Dict[str, Any], gemini_rubric: Dict[str, Any], 
                     user_prompt: str) -> Dict[str, Any]:
        """
        Merge rubrics from multiple LLMs using Claude as the synthesizer.
        
        Args:
            claude_rubric: Rubric from Claude
            gemini_rubric: Rubric from Gemini
            user_prompt: Original user requirements
            
        Returns:
            Final merged rubric
        """
        print("Merging rubrics from multiple LLMs...")
        
        merge_prompt = f"""You are synthesizing rubrics from multiple AI models to create the best possible evaluation framework.

**Original Requirements:**
{user_prompt}

**Claude's Rubric:**
{json.dumps(claude_rubric, indent=2)}

**Gemini's Rubric:**
{json.dumps(gemini_rubric, indent=2)}

**Task:**
Create a final, superior rubric that:
1. Combines the best criteria from both rubrics
2. Eliminates redundancy
3. Ensures comprehensive coverage
4. Maintains the same JSON structure
5. Keeps total points at 100 for each of Crackedness and Fit

Return the merged rubric as JSON with the same structure, adding:
{{
  "metadata": {{
    "focus_area": "brief description",
    "created_by": "ensemble",
    "generation_date": "{datetime.now().isoformat()}"
  }}
}}"""
        
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": merge_prompt}]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                rubric_json = response_text[json_start:json_end]
                return json.loads(rubric_json)
            else:
                raise ValueError("No valid JSON found in merge response")
                
        except Exception as e:
            print(f"Error merging rubrics: {e}")
            # Fallback to Claude's rubric
            return claude_rubric
    
    def generate_rubric(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main method to generate rubric using ensemble of LLMs.
        
        Args:
            user_prompt: Denny's specific requirements for the dinner
            
        Returns:
            Final comprehensive rubric
        """
        if not self.resumes:
            raise ValueError("No resumes loaded. Call load_resumes() first.")
        
        print("\n" + "="*60)
        print("RUBRIC GENERATION PROCESS")
        print("="*60 + "\n")
        
        # Create resume summary
        resume_summary = self.create_resume_summary()
        
        # Generate rubrics from multiple LLMs
        claude_rubric = self.generate_rubric_with_claude(user_prompt, resume_summary)
        gemini_rubric = self.generate_rubric_with_gemini(user_prompt, resume_summary)
        
        # Merge rubrics
        if claude_rubric and gemini_rubric:
            final_rubric = self.merge_rubrics(claude_rubric, gemini_rubric, user_prompt)
        elif claude_rubric:
            print("Warning: Using only Claude's rubric (Gemini failed)")
            final_rubric = claude_rubric
        elif gemini_rubric:
            print("Warning: Using only Gemini's rubric (Claude failed)")
            final_rubric = gemini_rubric
        else:
            raise Exception("Both LLMs failed to generate rubrics")
        
        self.rubric = final_rubric
        
        print("\n" + "="*60)
        print("RUBRIC GENERATION COMPLETE")
        print("="*60 + "\n")
        
        return final_rubric
    
    def save_rubric(self, output_path: str = "rubric.json"):
        """
        Save the generated rubric to a JSON file.
        
        Args:
            output_path: Path to save the rubric
        """
        if not self.rubric:
            raise ValueError("No rubric to save. Generate one first.")
        
        with open(output_path, 'w') as f:
            json.dump(self.rubric, f, indent=2)
        
        print(f"✓ Rubric saved to: {output_path}")
    
    def print_rubric_summary(self):
        """Print a human-readable summary of the rubric."""
        if not self.rubric:
            print("No rubric generated yet.")
            return
        
        print("\n" + "="*60)
        print("RUBRIC SUMMARY")
        print("="*60 + "\n")
        
        print("CRACKEDNESS CRITERIA (Total: 100 points)")
        print("-" * 60)
        for i, criterion in enumerate(self.rubric['crackedness_criteria'], 1):
            print(f"\n{i}. {criterion['name']} ({criterion['max_points']} points)")
            print(f"   {criterion['description']}")
        
        print("\n\nFIT CRITERIA (Total: 100 points)")
        print("-" * 60)
        for i, criterion in enumerate(self.rubric['fit_criteria'], 1):
            print(f"\n{i}. {criterion['name']} ({criterion['max_points']} points)")
            print(f"   {criterion['description']}")
        
        print("\n" + "="*60 + "\n")


def main():
    """
    Main function to run the rubric generator with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate rubric for VC+Founders Dinner applications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rubric_generator.py --prompt "Looking for founding engineers with startup experience"
  
  python rubric_generator.py --prompt "Healthcare founders at Series A stage" --output healthcare_rubric.json
  
  python rubric_generator.py --prompt "Infra/hard tech founders from big tech" --resume-dir ./applications
        """
    )
    
    parser.add_argument(
        '--prompt', 
        type=str, 
        required=True,
        help="Denny's requirements for this dinner (use quotes for multi-word prompts)"
    )
    
    parser.add_argument(
        '--resume-dir',
        type=str,
        default='resumes',
        help='Directory containing resumes (default: resumes)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='rubric.json',
        help='Output file for rubric (default: rubric.json)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VC+FOUNDERS DINNER RUBRIC GENERATOR")
    print("="*60)
    print(f"\nRequirements: {args.prompt}")
    print(f"Resume Directory: {args.resume_dir}")
    print(f"Output File: {args.output}\n")
    
    # Initialize generator
    generator = ResumeRubricGenerator(resume_dir=args.resume_dir)
    
    # Load all resumes
    generator.load_resumes()
    
    # Generate rubric
    rubric = generator.generate_rubric(args.prompt)
    
    # Print summary
    generator.print_rubric_summary()
    
    # Save to file
    generator.save_rubric(args.output)
    
    print("\nNext steps:")
    print(f"1. Review the rubric in {args.output}")
    print("2. Use this rubric to score each resume")
    print("3. Generate candidate rankings and descriptions")


if __name__ == "__main__":
    main()
