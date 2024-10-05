import { Component, OnInit, ViewChild, Output } from '@angular/core';
import {NgxChessBoardService} from 'ngx-chess-board';
import {NgxChessBoardView} from 'ngx-chess-board';
import { ApiService } from './api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  @ViewChild('board', { static: false })
  board!: NgxChessBoardView;

  turn = true;
  isGameFinished = false;
  endgameMessage = '';
  selectedOption = "lstm"
  isProcessingMove = false;
  selectedOptionText = 'LSTM Engine';
  uciPromotions = {
    "q": "1",
    "r": "2",
    "b": "3",
    "n": "4"
  }


  constructor(
    private ngxChessBoardService: NgxChessBoardService,
    private apiService: ApiService) {}

  ngOnInit(): void {}

  ngAfterViewInit() {
    this.board.reset();
  }

  switchPlayerTurn() {
    this.turn = !this.turn;
  }

  makeOponentsMove(move: string) {
    if (move.length == 5) {
      let translated = this.uciPromotions[move[4] as keyof typeof this.uciPromotions];
      let translatedMove = move.substring(0, 4) + translated;
      console.log(translatedMove);
      this.board.move(translatedMove);
    }
    else {
      this.board.move(move);
    }
  }

  MoveCompleted(event: any) {
    console.log(event)

    if (event.checkmate && this.turn) {
      this.endgameMessage = 'white won!';
      this.isGameFinished = true;
      console.warn(this.endgameMessage)
      return;
    }

    if (event.stalement) {
      this.endgameMessage = 'draw!';
      this.isGameFinished = true;
      console.warn(this.endgameMessage)
      return;
    }

    if (this.isProcessingMove) {
      return;
    }
    this.isProcessingMove = true;
    let fen = this.board.getFEN()
    console.log(fen);
    this.apiService.getData('get-move', this.selectedOption, fen).subscribe(
      response => {
        console.log('Data:', response);
        this.makeOponentsMove(response.move);
        this.isProcessingMove = false;
      },
      error => {
        console.error('Error:', error);
        this.isProcessingMove = false;
      }
    );
    this.switchPlayerTurn();
  }

  onResetGame() {
    this.board.reset();
    this.turn = true;
    this.endgameMessage = '';
    this.isGameFinished = false;
  }

  Reset() {
    this.board.reset();
    this.isGameFinished = false;
  }

  Reverse() {
    this.board.reverse()
  }

  handleOptionChange(option: string): void {
    this.selectedOption = option;
    console.log('Selected option:', this.selectedOption);
    switch (option) {
      case 'LSTM':
        this.selectedOptionText = 'LSTM Engine';
        break;
      case 'minimax':
        this.selectedOptionText = 'Minimax Algorithm';
        break;
      case 'alpha-beta':
        this.selectedOptionText = 'Alpha-Beta Pruning';
        break;
      default:
        this.selectedOptionText = 'Unknown Engine';
        break;
    }
  }
}