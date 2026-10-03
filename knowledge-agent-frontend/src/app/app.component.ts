import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from './services/api.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {

  question = '';
  answer = '';
  sources: { page: number; source: string }[] = [];
  loading = false;

  selectedFile: File | null = null;
  uploading = false;
  uploadMessage = '';
  documents: string[] = [];
selectedSource = '';

  constructor(private api: ApiService) {
  this.loadDocuments();
}
loadDocuments(): void {
  this.api.getDocuments().subscribe({
    next: (response) => {
      this.documents = response.documents;
    },
    error: (error) => {
      console.error('Erreur lors du chargement des documents', error);
    }
  });
}
  askQuestion(): void {

    if (!this.question.trim()) {
      return;
    }

    this.loading = true;
    this.answer = '';
    this.sources = [];

    this.api.askQuestion(
  this.question,
  this.selectedSource || undefined
).subscribe({
      next: (response) => {
        this.answer = response.answer;
        this.sources = response.sources;
        this.loading = false;
      },
      error: (error) => {
        console.error(error);

        this.answer =
          'Une erreur est survenue lors de la communication avec le serveur.';

        this.loading = false;
      }
    });
  }

  onFileSelected(event: Event): void {

    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.uploadMessage = '';
    }
  }

  uploadDocument(): void {

    if (!this.selectedFile) {
      return;
    }

    this.uploading = true;
    this.uploadMessage = '';

    this.api.uploadDocument(this.selectedFile).subscribe({
      next: (response) => {

        this.uploadMessage =
          `${response.filename} a été ajouté avec succès. ` +
          `${response.chunks_added} chunks indexés.`;

        this.uploading = false;
        const uploadedSource = `data/uploads/${response.filename}`;

this.selectedFile = null;
this.loadDocuments();
this.selectedSource = uploadedSource;
      },

      error: (error) => {

        console.error(error);

        this.uploadMessage =
          'Une erreur est survenue pendant l\'upload du document.';

        this.uploading = false;
      }
    });
  }
}