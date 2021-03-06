\def\filename{on.dtx}
\def\fileversion{v2.2}
\def\filedate{2011/02/08}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{on}[\filedate\ \fileversion\ ON Dissertations and Theses]
\LoadClass[12pt,a4paper,oneside]{book}
\RequirePackage[sort&compress]{natbib}
\RequirePackage{hyphenat}
\RequirePackage{lastpage}
\RequirePackage[english,brazil]{babel}
\RequirePackage{ifthen}
\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{tabularx}
\RequirePackage{eqparbox}
\RequirePackage{ltxcmds}
\RequirePackage[a4paper,bindingoffset=0.0cm,vcentering=true,%
top=2.5cm,bottom=2.5cm,left=3.0cm,right=3.0cm]{geometry}
\def\ONTeX{{\rm C\kern-.05em{\sc o\kern-.025em p\kern-.025em
p\kern-.025em e}}\kern-.08em
T\kern-.1667em\lower.5ex\hbox{E}\kern-.125emX\spacefactor1000}
\newboolean{maledoc}
\setboolean{maledoc}{false}
\newboolean{doutboolean}
\setboolean{doutboolean}{false}

% relatório feito durante o mestrado
\DeclareOption{reportm}{%
  \newcommand{\@degree}{Me.}
  \newcommand{\@degreename}{Mestrado}
  \newcommand{\local@degname}{Mestre}
  \newcommand{\foreign@degname}{Master}
  \setboolean{maledoc}{true}
  \newcommand\local@doctype{Relat\' orio}
  \newcommand\foreign@doctype{Report}
}

% relatório feito durante o doutorado
\DeclareOption{reportd}{%
  \newcommand{\@degree}{Dr.}
  \newcommand{\@degreename}{Doutorado}
  \newcommand{\local@degname}{Doutor}
  \newcommand{\foreign@degname}{Doctor}
  \setboolean{maledoc}{true}
  \newcommand\local@doctype{Relat\' orio}
  \newcommand\foreign@doctype{Report}
}

% exame de qualificação
\DeclareOption{dscexam}{%
  \newcommand{\@degree}{Dr.}
  \newcommand{\@degreename}{Doutorado}
  \newcommand{\local@degname}{Doutor}
  \newcommand{\foreign@degname}{Doctor}
  \setboolean{maledoc}{true}
  \newcommand\local@doctype{Relat\'{o}rio de Exame de Qualifica{\c c}{\~ a}o}
  \newcommand\foreign@doctype{Qualifying Exam Report}
}

% dissertação de mestrado
\DeclareOption{msc}{%
  \newcommand{\@degree}{M.Sc.}
  \newcommand{\@degreename}{Mestrado}
  \newcommand{\local@degname}{Mestre}
  \newcommand{\foreign@degname}{Master}
  \newcommand\local@doctype{Disserta{\c c}{\~ a}o}
  \newcommand\foreign@doctype{Dissertation}
}

% tese de doutorado
\DeclareOption{dsc}{%
  \newcommand{\@degree}{D.Sc.}
  \newcommand{\@degreename}{Doutorado}
  \newcommand{\local@degname}{Doutor}
  \newcommand{\foreign@degname}{Doctor}
  \newcommand\local@doctype{Tese}
  \newcommand\foreign@doctype{Thesis}
}

% projeto de pesquisa de mestrado
\DeclareOption{projm}{%
  \newcommand{\@degree}{Me.}
  \newcommand{\@degreename}{Mestrado}
  \newcommand{\local@degname}{Mestre}
  \newcommand{\foreign@degname}{Master}
  \setboolean{maledoc}{true}
  \newcommand\local@doctype{Projeto de pesquisa}
  \newcommand\foreign@doctype{Research project}
}

\DeclareOption{numbers}{%
  \PassOptionsToPackage{numbers}{natbib}
}


\onehalfspacing
\DeclareOption{doublespacing}{%
  \doublespacing
}

\ProcessOptions\relax
\newcommand\program[1]{%
  \ifthenelse{\equal{#1}{ASTRO}}
    {\global\def\local@progname{Astronomia}
     \global\def\foreign@progname{Astronomy}}{}
  \ifthenelse{\equal{#1}{GEO}}
    {\global\def\local@progname{Geof{\' i}sica}
     \global\def\foreign@progname{Geophysics}}{}
}

\renewcommand\title[1]{%
  \global\def\local@title{#1}%
}

\newcommand\foreigntitle[1]{%
  \global\def\foreign@title{#1}%
}


\newcommand\advisor[3]{%
  \global\def\@advtitle{#1}
  \global\def\@advname{#2}
  \global\def\@advsurn{#3}
}

\newcount\@coadvisor\@coadvisor0
\newcommand\coadvisor[3]{%
  \global\@namedef{ONcoAdvisorTitle:\expandafter\the\@coadvisor}{#1}
  \global\@namedef{ONcoAdvisorName:\expandafter\the\@coadvisor}{#2}
  \global\@namedef{ONcoAdvisorSurname:\expandafter\the\@coadvisor}{#3}
  \global\advance\@coadvisor by 1
  \ifnum\@coadvisor>1
    \renewcommand\local@coadvisorstring{Co-orientadores}
    \renewcommand\foreign@coadvisorstring{Co-advisors}
  \fi
}

\newcount\@examiner\@examiner0
\newcommand\examiner[3]{%
  \global\@namedef{ONExaminerTitle:\expandafter\the\@examiner}{#1}
  \global\@namedef{ONExaminerName:\expandafter\the\@examiner}{#2}
  \global\@namedef{ONExaminerSurname:\expandafter\the\@examiner}{#3}
  \global\advance\@examiner by 1
  \ifnum\@examiner>1
    \renewcommand\local@examinerstring{Examinadores}
    \renewcommand\foreign@examinerstring{Examiners}
  \fi
}

\renewcommand\author[2]{%
  \global\def\@authname{#1}
  \global\def\@authsurn{#2}
}

\renewcommand\date[2]{%
  \month=#1
  \year=#2
}

\newcommand\local@monthname{\ifcase\month\or
  Janeiro\or Fevereiro\or Mar{\c c}o\or Abril\or Maio\or Junho\or
  Julho\or Agosto\or Setembro\or Outubro\or Novembro\or Dezembro\fi}
  
\newcommand\foreign@monthname{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi}

\newcounter{keywords}

\newcommand\keyword[1]{%
  \global\@namedef{ONKeyword:\expandafter\the\c@keywords}{#1}
  \global\addtocounter{keywords}{1}
}


\pagenumbering{roman}
\setcounter{page}{2}

% begin folha de rosto

\renewcommand\frontmatter{%
  \cleardoublepage
  %\pagenumbering{roman}
  \thispagestyle{empty}
  %\setcounter{page}{2}
  \makefrontpage
  \clearpage
  \pagestyle{plain}
}

\renewcommand\maketitle{%
  \ltx@ifpackageloaded{hyperref}{\on@hypersetup}{}%
  \begin{titlepage}
  \begin{flushleft}
  \vspace*{1.5mm}
  \setlength\baselineskip{0pt}
  \setlength\parskip{1mm}
  \makebox[20mm][c]{\hspace{12.9cm}\includegraphics[width=5.0cm]{ON-logo}
  \hspace{6.0cm}\includegraphics[width=4.0cm]{DIPPG-logo}}
  \end{flushleft}
  \vspace{1.05cm}
  \begin{center}
  \nohyphens{\MakeUppercase\local@title}\par
  \vspace*{3cm}
  \nohyphens{\@authname\ \@authsurn}\par
  \end{center}
  \vspace*{2.1cm}
  \begin{flushright}
  \begin{minipage}{8.45cm}
  \frontcover@maintext
  \end{minipage}\par
  \vspace*{7.5mm}

  \nohyphens{%
  \begin{tabularx}{8.45cm}[b]{@{}l@{ }>{\raggedright\arraybackslash}X@{}}
    \local@advisorstring: & \@advtitle\ \@advname\ \@advsurn
        \vspace*{5.0mm} \\
    \ifnum\@coadvisor=0
    \else
    \local@coadvisorstring: &
    \fi
    \count1=0
    \toks@={}
    \@whilenum \count1<\@coadvisor \do{%
    \ifcase\count1 % same as \ifnum0=\count1
      \toks@=\expandafter{\csname ONcoAdvisorTitle:\the\count1%
            \expandafter\endcsname\expandafter\space%
        \csname ONcoAdvisorName:\the\count1%
        	\expandafter\endcsname\expandafter\space%
        \csname ONcoAdvisorSurname:\the\count1\endcsname\\}
    \else
      \toks@=\expandafter\expandafter\expandafter{%
        \expandafter\the\expandafter\toks@%
        \expandafter&\expandafter\space%
        \csname ONcoAdvisorTitle:\the\count1\expandafter\endcsname%
        \expandafter\space\csname ONcoAdvisorName:\the\count1\expandafter\endcsname%
        \expandafter\space\csname ONcoAdvisorSurname:\the\count1\endcsname\\
      }%
    \fi
    \advance\count1 by 1}
    \the\toks@
  \end{tabularx}}\par
  \end{flushright}
  \vspace*{\fill}
  \begin{center}
  \local@cityname\par
  \local@monthname\ de \number\year
  \end{center}
  \end{titlepage}
  \global\let\maketitle\relax%
  \global\let\and\relax}
\newcommand\makefrontpage{%
  \begin{center}
    \sloppy\nohyphens{\MakeUppercase\local@title}\par
    \vspace*{7mm}
    {\@authname\ \@authsurn}\par
  \end{center}\par
  \vspace*{4mm}
  \frontpage@maintext
  \vspace*{8mm}
  \noindent\local@approvedname:\par
     \begin{center}
  \begin{flushright}
        \count1=0
        \@whilenum \count1<\@examiner \do {
            \vskip26\p@ \baselineskip12pt
        \eqmakebox[signture@list]{\hrulefill}\\
        \eqmakebox[signture@list]{%
			 \ \ \ \csname ONExaminerTitle:\the\count1\expandafter\endcsname%
        		   \expandafter\space\csname ONExaminerName:\the\count1\endcsname%
        		   \expandafter\space\csname ONExaminerSurname:\the\count1\endcsname\ \ \ }\\
            \advance\count1 by 1}
  \end{flushright}
     \end{center}
  \vspace*{\fill}
  \frontpage@bottomtext}
\newcommand\on@hypersetup{%
\begingroup
  % changes to \toks@ and \count@ are kept local;
  % it's not necessary for them, but it is usually the case
  % for \count1, because the first ten counters are written
  % to the DVI file, thus you got lucky because of PDF output
  \toks@={}% in this special case not necessary
  \count@=0 %
  \@whilenum\count@<\value{keywords}\do{%
    % * a keyword separator is not necessary,
    %    if there is just one keyword
    % * \csname ONKeyword:\the\count@\endcsname must be expanded
    %    at least once, to get rid of the loop depended \count@
    \ifcase\count@ % same as \ifnum0=\count@
      \toks@=\expandafter{\csname ONKeyword:\the\count@\endcsname}%
    \else
      \toks@=\expandafter\expandafter\expandafter{%
        \expandafter\the\expandafter\toks@
        \expandafter;\expandafter\space
        \csname ONKeyword:\the\count@\endcsname
      }%
    \fi
    \advance\count@ by 1 %
  }%
  \edef\x{\endgroup
    \noexpand\hypersetup{%
      pdfkeywords={\the\toks@}%
    }%
  }%
\x
\hypersetup{%
  pdfauthor={\@authname\ \@authsurn},
  pdftitle={\local@title},
  pdfsubject={\local@doctype\ de \@degreename\ em \local@progname\ da ON},
  pdfcreator={LaTeX with ONTeX toolkit},
  breaklinks={true},
  raiselinks={true},
  pageanchor={true},
}}

% end folha de rosto

% begin ficha catalográfica

\newcommand\makecatalog{%
  \vspace*{\fill}
  \begin{center}
    \setlength{\fboxsep}{5mm}
    \framebox[120mm][c]{\makebox[5mm][c]{}%
      \begin{minipage}[c]{105mm}
      \setlength{\parindent}{5mm}
      \noindent\sloppy\nohyphens\@authsurn,
      \nohyphens\@authname\par
      \nohyphens{\local@title/\@authname\ \@authsurn. -- \local@cityname:
      ON, \number\year.}\par
      \pageref{front:pageno},
      \pageref{LastPage}
      p.\@ifundefined{r@cat:lofflag}{}{\pageref{cat:lofflag}} $29,7$cm.\par
      % There is an issue here. When the last entry must be split between lines,
      % the spacing between it and the next paragraph becomes smaller.
      % Should we manually introduce a fixed space? But how could we know that
      % a name was split? Is this happening yet?
      \nohyphens{%
      \begin{tabularx}{100mm}[b]{@{}l@{ }>{\raggedright\arraybackslash}X@{}}
   \local@advisorstring: & \@advname\ \@advsurn \\
        \ifnum\@coadvisor=0
        \else
        \local@coadvisorstring: &
        \fi
        \count1=0
        \toks@={}
        \@whilenum \count1<\@coadvisor \do{%
        \ifcase\count1 % same as \ifnum0=\count1
          \toks@=\expandafter{\csname ONcoAdvisorName:\the\count1%
          \expandafter\endcsname\expandafter\space%
          \csname ONcoAdvisorSurname:\the\count1\endcsname\\}
        \else
          \toks@=\expandafter\expandafter\expandafter{%
            \expandafter\the\expandafter\toks@
            \expandafter&\expandafter\space
            \csname ONcoAdvisorName:\the\count1\expandafter\endcsname%
            \expandafter\space\csname ONcoAdvisorSurname:\the\count1\endcsname\\
          }%
        \fi
        \advance\count1 by 1}
        \the\toks@
      \end{tabularx}}\par
      \nohyphens{\local@doctype\ ({\MakeLowercase\@degreename}) --
      ON/Programa de P{\' o}s-gradua{\c c}{\~ a}o em \local@progname,
      \number\year.}\par
      \bibname: p. \pageref{bib:begin} -- \pageref{bib:end}.\par
      \count1=0
      \count2=1
      \nohyphens{\@whilenum \count1<\value{keywords} \do {%
        \number\count2. \csname ONKeyword:\the\count1 \endcsname.
        \advance\count1 by 1
        \advance\count2 by 1}
      I. \csname ONcoAdvisorSurname:0\endcsname,%
      \ \csname ONcoAdvisorName:0\endcsname%
      \ifthenelse{\@coadvisor>1}{\ \emph{et~al.}{}}.
      II. \local@universityname, Programa de P{\' o}s-gradua{\c c}{\~ a}o
      em \local@progname.
      III. T{\' i}tulo.}
    \end{minipage}}
  \end{center}
  \vspace*{\fill}}

% end ficha catalográfica

\newcommand\dedication[1]{
  \gdef\@dedic{#1}
    \cleardoublepage
    \vspace*{\fill}
    \begin{flushright}
      \begin{minipage}{60mm}
        \raggedleft \it \normalsize \@dedic
      \end{minipage}
    \end{flushright}}

\newenvironment{abstract}{%
  \clearpage
  \thispagestyle{plain}
  \abstract@toptext\par
  \vspace*{8.6mm}
  \begin{center}
    \sloppy\nohyphens{\MakeUppercase\local@title}\par
    \vspace*{13.2mm}
    \@authname\ \@authsurn \par
    \vspace*{7mm}
    \local@monthname/\number\year
  \end{center}\par

  \vspace*{7mm}}{\vspace*{\fill}}
\newenvironment{foreignabstract}{%
  \clearpage
  \thispagestyle{plain}
  \begin{otherlanguage}{english}
  \foreignabstract@toptext\par
  \vspace*{8.6mm}
  \begin{center}
    \sloppy\nohyphens{\MakeUppercase\foreign@title}\par
    \vspace*{13.2mm}
    \@authname\ \@authsurn \par
    \vspace*{7mm}
    \foreign@monthname/\number\year
  \end{center}\par
  \vspace*{7mm}}{%
  \end{otherlanguage}
  \vspace*{\fill}
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\foreign@title\@empty
  \global\let\foreign@title\relax
  \global\let\local@title\@empty
  \global\let\local@title\relax
  \global\let\author\relax
  \global\let\author\relax
  \global\let\date\relax}
  

\renewcommand\listoffigures{%
    \on@hasLof
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename}%
      \addcontentsline{toc}{chapter}{\listfigurename}%
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }

\renewcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename}%
      \addcontentsline{toc}{chapter}{\listtablename}%
      \@mkboth{%
          \MakeUppercase\listtablename}%
         {\MakeUppercase\listtablename}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
    
   
% begin symbols e abbreviations

\newcommand\printlosymbols{%
\renewcommand\glossaryname{\listsymbolname}%
\@input@{\jobname.los}}
\def\makelosymbols{%
  \newwrite\@losfile
  \immediate\openout\@losfile=\jobname.syx
  \def\symbl{\@bsphack\begingroup
             \@sanitize
             \@wrlos}\typeout
  {Writing index of symbols file \jobname.syx}%
  \let\makelosymbols\@empty
}

\@onlypreamble\makelosymbols
\AtBeginDocument{%
\@ifpackageloaded{hyperref}{%
  \def\@wrlos#1#2{%
    \protected@write\@losfile{}%
      {\string\indexentry{[#1] #2|hyperpage}{\thepage}}%
    \endgroup
    \@esphack}}{%
  \def\@wrlos#1#2{%
    \protected@write\@losfile{}%
      {\string\indexentry{[#1] #2}{\thepage}}%
    \endgroup
    \@esphack}}}

\newcommand\printloabbreviations{%
\renewcommand\glossaryname{\listabbrevname}%
\@input@{\jobname.lab}}
\def\makeloabbreviations{%
  \newwrite\@labfile
  \immediate\openout\@labfile=\jobname.abx
  \def\abbrev{\@bsphack\begingroup
              \@sanitize
              \@wrlab}\typeout
  {Writing index of abbreviations file \jobname.abx}%
  \let\makeloabbreviations\@empty
}

\@onlypreamble\makeloabbreviations
\AtBeginDocument{%
\@ifpackageloaded{hyperref}{%
  \def\@wrlab#1#2{%
    \protected@write\@labfile{}%
      {\string\indexentry{[#1] #2|hyperpage}{\thepage}}%
    \endgroup
    \@esphack}}{%
  \def\@wrlab#1#2{%
    \protected@write\@labfile{}%
      {\string\indexentry{[#1] #2}{\arabic{page}}}%
    \endgroup
    \@esphack}}}
\AtBeginDocument{%
  \ltx@ifpackageloaded{hyperref}{
    \def\on@bibEnd{%
      \immediate\write\@auxout{%
        \string\newlabel{bib:end}{{}{\arabic{page}}{}{page.\arabic{page}}{}}}}%
    \def\on@bibBegin{%
      \immediate\write\@auxout{%
        \string\newlabel{bib:begin}{{}{\arabic{page}}{}{page.\arabic{page}}{}}}}%
    \def\on@mainBegin{%
      \immediate\write\@auxout{%
        \string\newlabel{front:pageno}{{}{\Roman{page}}{}{page.\roman{page}}{}}}}%
    \def\on@hasLof{%
      \immediate\write\@auxout{%
        \string\newlabel{cat:lofflag}{{}{:~il.;}{}{page.\roman{page}}{}}}}%
  }{%
    \def\on@bibEnd{%
      \immediate\write\@auxout{%
      \string\newlabel{bib:end}{{}{\arabic{page}}{}}}}%
    \def\on@bibBegin{%
      \immediate\write\@auxout{%
      \string\newlabel{bib:begin}{{}{\arabic{page}}{}}}}%
    \def\on@mainBegin{%
      \immediate\write\@auxout{%
        \string\newlabel{front:pageno}{{}{\Roman{page}}{}}}}%
    \def\on@hasLof{%
      \immediate\write\@auxout{%
        \string\newlabel{cat:lofflag}{{}{:~il.;}{}}}}%
  }%
}

% end symbols e abbreviations

\newdimen\bibindent%
\setlength\bibindent{1.5em}%
\renewenvironment{thebibliography}[1]%
     {\onehalfspacing%
      \chapter*{\bibname}%
      \addcontentsline{toc}{chapter}{\bibname}%
      \on@bibBegin
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\setlength{\labelwidth}{0ex}%
            \setlength{\leftmargin}{9.0ex}%
            \setlength{\itemindent}{-9.0ex}%
            \advance\leftmargin\labelsep%
            \@openbib@code%
            \usecounter{enumiv}%
            \let\p@enumiv\@empty%
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy%
      \clubpenalty4000%
      \@clubpenalty \clubpenalty%
      \widowpenalty4000%
      \sfcode`\.\@m}%
     {\def\@noitemerr%
       {\@latex@warning{Empty `thebibliography' environment}}%
       \on@bibEnd
      \endlist}

\newenvironment{theglossary}{%
  \if@twocolumn%
    \@restonecoltrue\onecolumn%
  \else%
    \@restonecolfalse%
  \fi%
  \@mkboth{\MakeUppercase\glossaryname}%
  {\MakeUppercase\glossaryname}%
  \chapter*{\glossaryname}%
  \addcontentsline{toc}{chapter}{\glossaryname}
  \list{}
  {\setlength{\listparindent}{0in}%
   \setlength{\labelwidth}{1.0in}%
   \setlength{\leftmargin}{1.5in}%
   \setlength{\labelsep}{0.5in}%
   \setlength{\itemindent}{0in}}%
   \sloppy}%
  {\if@restonecol\twocolumn\fi%
\endlist}

\renewenvironment{theindex}{%
  \if@twocolumn
    \@restonecolfalse
  \else
    \@restonecoltrue
  \fi
  \twocolumn[\@makeschapterhead{\indexname}]%
  \@mkboth{\MakeUppercase\indexname}%
  {\MakeUppercase\indexname}%
  \thispagestyle{plain}\parindent\z@
  \addcontentsline{toc}{chapter}{\indexname}
  \parskip\z@ \@plus .3\p@\relax
  \columnseprule \z@
  \columnsep 35\p@
  \let\item\@idxitem}
  {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\glossaryname{Glossary}
%\newcommand\listabbreviationname{Lista de Abreviaturas}
%\newcommand\listsymbolname{Lista de S{\' i}mbolos}
\newcommand\local@advisorstring{Orientador(a)}
\newcommand\foreign@advisorstring{Advisor}
\newcommand\local@coadvisorstring{Co-orientador(a)}
\newcommand\foreign@coadvisorstring{Co-advisor}
\newcommand\local@examinerstring{Examinador(a)}
\newcommand\foreign@examinerstring{Examiner}
\ifthenelse{\boolean{maledoc}}{%
  \newcommand\local@approvedname{Examinado por}%
}{%
  \newcommand\local@approvedname{Examinada por}%
}
\newcommand\local@universityname{Observat{\' o}rio Nacional}
\newcommand\local@progstring{Programa}
\newcommand\foreign@progstring{Program}
\newcommand\local@cityname{Rio de Janeiro}
\newcommand\local@statename{RJ}
\newcommand\local@countryname{Brasil}
\newcommand\frontcover@maintext{
  \sloppy\nohyphens{\local@doctype\
  \ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}
  ao Programa de P{\' o}s-gradua{\c c}{\~ a}o em \local@progname\ do
  \local@universityname, como parte dos requisitos
  necess{\' a}rios {\` a} obten{\c c}{\~ a}o do \ifthenelse{\boolean{doutboolean}}{grau}{t{\' i}tulo}
  de \local@degname\ em \local@progname.}
}
\newcommand\frontpage@maintext{
  \noindent {\MakeUppercase\local@doctype}
  \ifthenelse{\boolean{maledoc}}{SUBMETIDO}{SUBMETIDA}
  \sloppy\nohyphens{AO PROGRAMA DE P{\' O}S-GRADUA{\c C}{\~ A}O EM
  {\MakeUppercase\local@progname} DO OBSERVAT{\' O}RIO
  NACIONAL COMO PARTE DOS REQUISITOS
  NECESS{\' A}RIOS PARA A OBTEN{\c C}{\~ A}O DO
  \ifthenelse{\boolean{doutboolean}}{GRAU}{T{\' I}TULO} DE
  {\MakeUppercase\local@degname} EM 
  {\MakeUppercase\local@progname.\par}}%
}
\newcommand\frontpage@bottomtext{%
  \begin{center}
  {\MakeUppercase{\local@cityname, \local@statename\ -- \local@countryname}}\par
  {\MakeUppercase\local@monthname\ DE \number\year}
  \end{center}%
}
\newcommand\abstract@toptext{%
  \noindent Resumo \ifthenelse{\boolean{maledoc}}{do}{da}
  \local@doctype\ \ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}
  \sloppy\nohyphens{ao Programa de P{\' o}s-Gradua{\c c}{\~ a}o em \local@progname\ do Observat{\' o}rio Nacional como parte dos requisitos
  necess{\' a}rios para a obten{\c c}{\~ a}o do
  \ifthenelse{\boolean{doutboolean}}{grau}{t{\' i}tulo} de
  \local@degname\ em {\local@progname.\par}}
}
\newcommand\foreignabstract@toptext{%
  \noindent Abstract of the
  \foreign@doctype\ presented
  \sloppy\nohyphens{to the National Observatory's Graduate Program in \foreign@progname\ as a partial fulfillment
  of the requirements for the degree of \foreign@degname\ in {\foreign@progname.\par}}
}
