import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  ApiService,
  AskTurn,
  describeError,
  DocumentInfo,
  SourceRef,
  StreamEvent,
} from './services/api.service';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceRef[];
  answerable?: boolean;
  tool?: string;
  rewritten?: string | null;
  streaming?: boolean;
  stopped?: boolean;
  error?: string;
}

const MAX_HISTORY_MESSAGES = 8;

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  @ViewChild('anchor') private anchor?: ElementRef<HTMLElement>;

  documents: DocumentInfo[] = [];
  selectedSource = '';
  documentsError = '';

  messages: ChatMessage[] = [];
  question = '';
  asking = false;
  private abort?: AbortController;

  selectedFile: File | null = null;
  uploading = false;
  uploadMessage = '';
  uploadFailed = false;

  constructor(readonly api: ApiService) {}

  ngOnInit(): void {
    this.loadDocuments();
  }

  loadDocuments(): void {
    this.api.getDocuments().subscribe({
      next: (response) => {
        this.documents = response.documents;
        this.documentsError = '';
        if (this.selectedSource && !this.documents.some((d) => d.id === this.selectedSource)) {
          this.selectedSource = '';
        }
      },
      error: (err) => (this.documentsError = describeError(err)),
    });
  }

  // --- Documents --------------------------------------------------------- //
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile = input.files?.[0] ?? null;
    this.uploadMessage = '';
  }

  uploadDocument(fileInput: HTMLInputElement): void {
    if (!this.selectedFile) {
      return;
    }
    this.uploading = true;
    this.uploadMessage = '';
    this.api.uploadDocument(this.selectedFile).subscribe({
      next: (response) => {
        this.uploading = false;
        this.uploadFailed = false;
        this.uploadMessage = `${response.name} ajouté : ${response.pages} pages, ${response.chunks_added} passages indexés.`;
        this.selectedFile = null;
        fileInput.value = '';
        this.loadDocuments();
        this.selectedSource = response.id;
      },
      error: (err) => {
        this.uploading = false;
        this.uploadFailed = true;
        this.uploadMessage = describeError(err);
      },
    });
  }

  deleteDocument(doc: DocumentInfo): void {
    if (!confirm(`Supprimer « ${doc.name} » et ses ${doc.chunks} passages indexés ?`)) {
      return;
    }
    this.api.deleteDocument(doc.id).subscribe({
      next: () => this.loadDocuments(),
      error: (err) => {
        this.uploadFailed = true;
        this.uploadMessage = describeError(err);
      },
    });
  }

  // --- Conversation ------------------------------------------------------- //
  async ask(): Promise<void> {
    const question = this.question.trim();
    if (!question || this.asking) {
      return;
    }

    const history: AskTurn[] = this.messages
      .filter((m) => !m.error && m.content)
      .slice(-MAX_HISTORY_MESSAGES)
      .map((m) => ({ role: m.role, content: m.content }));

    const reply: ChatMessage = { role: 'assistant', content: '', sources: [], streaming: true };
    this.messages.push({ role: 'user', content: question }, reply);
    this.question = '';
    this.asking = true;
    this.abort = new AbortController();
    this.scrollDown();

    try {
      await this.api.askStream(
        { question, source: this.selectedSource || null, history },
        (event) => this.applyEvent(reply, event),
        this.abort.signal,
      );
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') {
        reply.stopped = true;
      } else {
        reply.error = describeError(err);
      }
    } finally {
      reply.streaming = false;
      this.asking = false;
      this.abort = undefined;
      this.scrollDown();
    }
  }

  stop(): void {
    this.abort?.abort();
  }

  clearConversation(): void {
    this.stop();
    this.messages = [];
  }

  onEnter(event: Event): void {
    const e = event as KeyboardEvent;
    if (!e.shiftKey) {
      e.preventDefault();
      void this.ask();
    }
  }

  private applyEvent(reply: ChatMessage, event: StreamEvent): void {
    switch (event.type) {
      case 'meta':
        reply.sources = event.sources;
        reply.answerable = event.answerable;
        reply.tool = event.tool;
        reply.rewritten = event.rewritten_question;
        break;
      case 'token':
        reply.content += event.text;
        this.scrollDown();
        break;
      case 'error':
        reply.error = event.detail;
        break;
    }
  }

  private scrollDown(): void {
    queueMicrotask(() => this.anchor?.nativeElement.scrollIntoView?.({ block: 'end' }));
  }
}
